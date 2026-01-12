from math import exp
import tonic
from tonic import MemoryCachedDataset, SlicedDataset
from tonic.slicers import SliceByTime
from pytorch_lightning import LightningDataModule
from tonic.transforms import NumpyAsType
from torch.utils.data import DataLoader, random_split
import torch
from torchvision import transforms
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


LABEL_TO_NAME = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
}


@dataclass(frozen=True)
class ToVoxelGrid:
    sensor_size: tuple[int, int, int]
    n_time_bins: int

    def __call__(self, in_events):
        events = in_events.copy()
        assert self.sensor_size[2] == 2

        voxel_grid = np.zeros((self.n_time_bins, self.sensor_size[1], self.sensor_size[0]), float).ravel()

        # normalize the event timestamps so that they lie between 0 and n_time_bins
        try:
            ts = (
                self.n_time_bins
                * (events["t"].astype(float) - events["t"][0])
                / (events["t"][-1] - events["t"][0])
            )
        except IndexError:
            print(events)

        xs = events["x"].astype(int)
        ys = events["y"].astype(int)
        pols = events["p"]
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = ts.astype(int)
        dts = ts - tis
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts

        valid_indices = tis < self.n_time_bins
        np.add.at(
            voxel_grid,
            xs[valid_indices]
            + ys[valid_indices] * self.sensor_size[0]
            + tis[valid_indices] * self.sensor_size[0] * self.sensor_size[1],
            vals_left[valid_indices],
        )

        valid_indices = (tis + 1) < self.n_time_bins
        np.add.at(
            voxel_grid,
            xs[valid_indices]
            + ys[valid_indices] * self.sensor_size[0]
            + (tis[valid_indices] + 1) * self.sensor_size[0] * self.sensor_size[1],
            vals_right[valid_indices],
        )

        voxel_grid = np.reshape(
            voxel_grid, (self.n_time_bins, 1, self.sensor_size[1], self.sensor_size[0])
        )

        return voxel_grid


class ToDict:
    def __call__(self, events):
        x = {"x": events["x"], 
             'y': events["y"],
             't': events["t"],
             'p': events["p"]
             }
        return x

class ToTorchTensor:
    def __call__(self, x):
        x = torch.from_numpy(x).float()

        if len(x.shape) == 3:
            return x

        T, C, H, W = x.shape
        return x.reshape(T * C, H, W)

class Torch:
    def __call__(self, x):
        x = torch.from_numpy(x).float()
        T, C, H, W = x.shape
        return x.reshape(T * C, H, W)

class CIFAR10Datamodule(LightningDataModule):
    def __init__(self, batch_size=16, num_workers=5, resize_size=128, 
                 time_step=50_000, n_bins=5):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_size = resize_size
        self.time_step = time_step
        self.n_bins = n_bins
        self.sensor_size = tonic.datasets.CIFAR10DVS.sensor_size


        dataset = tonic.datasets.CIFAR10DVS(save_to="./data/")
        self.train_data, self.val_data, self.test_data = random_split(dataset, [0.7, 0.15, 0.15]) # type: ignore

        # frame_transform = tonic.transforms.ToFrame(sensor_size=tonic.datasets.CIFAR10DVS.sensor_size,
        #                                            n_time_bins=n_bins,
        #                                            include_incomplete=True,
        #                                            overlap=self.overlap)

        frame_transform = ToVoxelGrid(sensor_size=self.sensor_size, n_time_bins=n_bins)

        self.train_transform = transforms.Compose([
            # tonic.transforms.Denoise(filter_time=10e3),
            ToDict(),
            frame_transform,
            ToTorchTensor(),
            v2.ToDtype(torch.float32, scale=True),
            transforms.Resize((self.resize_size, self.resize_size), 
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip()
            ])

        self.test_transform = transforms.Compose([
            # tonic.transforms.Denoise(filter_time=10e3),
            ToDict(),
            frame_transform,
            ToTorchTensor(),
            v2.ToDtype(torch.float32, scale=True),
            transforms.Resize((self.resize_size, self.resize_size), 
                               interpolation=transforms.InterpolationMode.NEAREST),
            ])

    def setup(self, stage=None): 
        print(f"Slice time: {self.time_step}")
        slicer = SliceByTime(time_window=self.time_step, include_incomplete=False)
        self.train_ds = MemoryCachedDataset(SlicedDataset(self.train_data, slicer=slicer, # type: ignore
                                       metadata_path="./metadata/cifar10/train",
                                       transform=self.train_transform))

        self.val_ds = MemoryCachedDataset(SlicedDataset(self.val_data, slicer=slicer, # type: ignore
                                       metadata_path="./metadata/cifar10/val",
                                       transform=self.test_transform))

        self.test_ds = MemoryCachedDataset(SlicedDataset(self.test_data, slicer=slicer, # type: ignore
                                       metadata_path="./metadata/cifar10/test",
                                       transform=self.test_transform))

    def train_dataloader(self):
        return DataLoader(self.train_ds, # type: ignore
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=tonic.collation.PadTensors()
                          )

    def val_dataloader(self):
        return DataLoader(self.val_ds, # type: ignore
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=tonic.collation.PadTensors()
                          )

    def test_dataloader(self):
        return DataLoader(self.test_ds, # type: ignore
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=tonic.collation.PadTensors() # TODO: check if needed
                          )

def plot_frames(frames):
    fig, axes = plt.subplots(1, len(frames))
    for axis, frame in zip(axes, frames):
        axis.imshow(frame[1] - frame[0])
        axis.axis("off")
    plt.tight_layout()


def main():
    dm = CIFAR10Datamodule()
    dm.setup()

    print("Train dataset length:", len(dm.train_ds))
    print("Val dataset length:", len(dm.val_ds))
    print("Test dataset length:", len(dm.test_ds))

    img, label = dm.train_ds[0]
    print(img.shape)

    for i in img:
        plt.imshow(i)
        plt.title(LABEL_TO_NAME[label])
        plt.show()


    for ds in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        for img, label in dm.train_ds:
            for i in img:
                s = torch.sum(i)
                assert s != 0

    # plot_frames(img)


if __name__ == "__main__":
    main()

from functools import cache
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

import tonic
from tonic import CachedDataset
from tonic.slicers import SliceByTime
from tonic.sliced_dataset import SlicedDataset

import matplotlib.pyplot as plt


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


class ToTorchTensor:
    def __call__(self, x):
        x = torch.from_numpy(x).float().clone()

        if len(x.shape) == 3:
            return x

        T, C, H, W = x.shape
        return x.reshape(T * C, H, W)


class TonicDatsetWithIdx:
    def __init__(self, dataset, video_id):
        self.dataset = dataset
        self.video_id = video_id

    def __len__(self):
        return len(self.video_id)

    def __getitem__(self, idx):
        video_id = self.video_id[idx]
        return self.dataset[video_id]


class CIFAR10Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        n_time_bins: int = 10,
        frame_time_us: int = 100_000,   # 10 ms
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.n_time_bins = n_time_bins
        self.frame_time_us = frame_time_us
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage=None):
        base_ds = tonic.datasets.CIFAR10DVS(save_to=self.data_dir)

        # Split videos before slicing 
        all_ids = list(range(len(base_ds)))
        all_targets = [base_ds.targets[id] for id in all_ids]

        # even split of all classes 
        id_train, id_val_test, _, target_val_test = train_test_split(all_ids, all_targets, test_size=0.3, random_state=42, stratify=all_targets)
        id_val, id_test, _, _ = train_test_split(id_val_test, target_val_test, test_size=0.5, random_state=42, stratify=target_val_test)


        slicer = SliceByTime(
            time_window=self.frame_time_us,
            overlap=0,
            include_incomplete=False,
        )


        to_frame = tonic.transforms.ToVoxelGrid(
                sensor_size=tonic.datasets.CIFAR10DVS.sensor_size,
                n_time_bins=self.n_time_bins,)

        # to_frame = tonic.transforms.ToFrame(
        #         sensor_size=tonic.datasets.CIFAR10DVS.sensor_size,
        #         n_time_bins=self.n_time_bins,)

        # to_frame = tonic.transforms.ToTimesurface(
        #     sensor_size=tonic.datasets.CIFAR10DVS.sensor_size,
        #     dt=10_000,
        #     tau=15_000,
        # )

        train_transform = transforms.Compose([
            to_frame,
            ToTorchTensor(),
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST),
            #transforms.RandomRotation(10),
            ]
           )
        test_transform = transforms.Compose([
            to_frame,
            ToTorchTensor(),
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST),
            ]
       )

        train_aa = TonicDatsetWithIdx(base_ds, id_train)
        self.train_sliced_ds = SlicedDataset(train_aa, # type: ignore 
                                            slicer=slicer, # type: ignore 
                                            transform=train_transform, 
                                            )


        val_aa = TonicDatsetWithIdx(base_ds, id_val)

        self.val_sliced_ds = SlicedDataset(val_aa, # type: ignore
                                      slicer=slicer, # type: ignore
                                      transform=test_transform,
                                      )

        test_aa = TonicDatsetWithIdx(base_ds, id_test)
        self.test_sliced_ds = SlicedDataset(test_aa, # type: ignore
                                       slicer=slicer, # type: ignore
                                       transform=test_transform,
                                       )

        print(f"train_sliced:{len(self.train_sliced_ds)}")
        print(f"val_sliced:{len(self.val_sliced_ds)}")
        print(f"test_sliced:{len(self.test_sliced_ds)}")
        


    def train_dataloader(self):
        return DataLoader(
            self.train_sliced_ds, # type: ignore
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
        )


    def val_dataloader(self):
        return DataLoader(
            self.val_sliced_ds, # type: ignore
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
                self.test_sliced_ds, # type: ignore
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=self.num_workers,
            )

def main():
    dm = CIFAR10Datamodule()
    dm.setup()

    print("Train dataset length:", len(dm.train_sliced_ds))
    print("Val dataset length:", len(dm.val_sliced_ds))
    print("Test dataset length:", len(dm.test_sliced_ds))

    # img, label = dm.train_sliced_ds[1]
    img, label = dm.train_sliced_ds[1]
    print(img.shape)

    for i in img:
        plt.imshow(i[0, :, :])
        plt.title(LABEL_TO_NAME[label])
        plt.show()


    for ds in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        for img, label in ds:
            s = torch.sum(img)
            assert s != 0

    # plot_frames(img)


if __name__ == "__main__":
    main()

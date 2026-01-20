from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms
from pathlib import Path
import numpy as np
import tonic


NAME_TO_LABEL = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}


def get_video_id(file_name: str):
    video_id = str(file_name).split('_')[2]
    return video_id


class CIFAR10DVSBinsAeadat(Dataset):
    '''
    Item shape: [sample_bins, polarity_channels, sensor_size[0], sensor_size[1]]
    '''
    def __init__(
        self,
        video_ids: dict[int, set[str]],
        root: Path,
        sample_bins: int = 1,
        time_step: int = 50_000,
        sensor_size=(128, 128),
        pos_polarity=True,
        neg_polarity=True,
        max_events_per_px=5,
        transform=None,
        event_transform=None,
        num_classes=10,
        join_channels=True
    ):
        self.root = root
        self.video_ids = video_ids
        self.sample_bins = sample_bins
        self.time_step = time_step
        self.sensor_size = sensor_size
        self.pos_polarity = pos_polarity
        self.neg_polarity = neg_polarity
        self.max_events_per_px = max_events_per_px
        self.transform = transform
        self.event_transform = event_transform
        self.num_classes = num_classes
        self.join_channels = join_channels

        # (file_path, label, start_bin)
        self.samples = []

        for class_folder in sorted(self.root.iterdir()):
            if not class_folder.is_dir():
                continue

            if class_folder.name in NAME_TO_LABEL:
                label = NAME_TO_LABEL[class_folder.name]
            else:
                label = int(class_folder.name)

            for f in class_folder.iterdir():
                if f.suffix != ".aedat4":
                    continue

                video_id = f.stem.split('_')[2]
                if video_id not in self.video_ids[label]:
                    continue

                events = np.array(tonic.io.read_aedat4(str(f)))
                t = events['t']
                t_min, t_max = t.min(), t.max()
                n_bins = int(np.floor((t_max - t_min) / self.time_step))

                for start_bin in range(0, n_bins):
                    self.samples.append((f, label, start_bin))

        print(
            f"Loaded {len(self.samples)} samples "
            f"({sample_bins} bins per sample)"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, start_bin = self.samples[idx]

        events = np.array(tonic.io.read_aedat4(str(path)))
        t = events['t']
        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)
        p = events['p'].astype(np.float32)

        if self.event_transform:
            t, x, y, p = self.event_transform(t, x, y, p)

        K = self.sample_bins
        H, W = self.sensor_size
        C = int(self.pos_polarity) + int(self.neg_polarity)

        sample_idx = ((t - t.min()) // self.time_step).astype(np.int32)

        mask = (sample_idx == start_bin)
        t = t[mask]
        x = x[mask]
        y = y[mask]
        p = p[mask]

        p_pos = (p == 1)
        p_neg = (p == -1)

        bin_time_step = self.time_step / K
        bin_idx = ((t - t.min()) // bin_time_step).astype(int)

        x = (x * W) // 128
        y = (y * H) // 128

        voxel = np.zeros((K, C, H, W), dtype=np.float32)

        c: int = 0
        if self.pos_polarity:
            np.add.at(voxel, (bin_idx, c, y, x), p_pos) # type: ignore
            c += 1

        if self.neg_polarity:
            np.add.at(voxel, (bin_idx, c, y, x), p_neg) # type: ignore

        np.clip(voxel, 0, self.max_events_per_px, out=voxel)

        voxel = torch.tensor(voxel)
        if self.transform:
            voxel = self.transform(voxel)

        if self.join_channels:
            voxel = voxel.reshape(-1, voxel.shape[2], voxel.shape[3]) # type: ignore

        # standarize values to [0, 1.0]
        max = voxel.max() 
        min = voxel.min()
        voxel =  (voxel - min) / (max - min)

        return voxel, label
    

class CIFAR10DVSAedatDataModule(LightningDataModule):
    '''
    Batch shape: [batch_size, sample_bins, polarity_channels, 64, 64]
    '''
    def __init__(
        self,
        batch_size=12,
        num_workers=4,
        val_split=0.15,
        test_split=0.15,
        sensor_size=(128, 128),
        time_step=50_000,
        sample_bins=1,
        root: Path = Path("./data/CIFAR10DVS"),
        num_classes=10,
        pos_polarity=True,
        neg_polarity=True,
        max_events_per_px=5, 
        join_channels=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.sensor_size = sensor_size
        self.time_step = time_step
        self.sample_bins = sample_bins
        self.root = root
        self.num_classes = num_classes
        self.pos_polarity = pos_polarity
        self.neg_polarity = neg_polarity
        self.max_events_per_px = max_events_per_px
        self.join_channels = join_channels

        # Transformacje na voxel grid
        self.train_transform = transforms.Compose([
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST),
            #transforms.Normalize(mean=[0], std=[1]),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(10)
        ])
        self.val_test_transform = transforms.Compose([
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST),
            #transforms.Normalize(mean=[0], std=[1]),
        ])

        self._compute_split()

    def _compute_split(self):
        # zbierz video_id dla każdej klasy
        video_per_label: dict[int, list[str]] = {i: [] for i in range(self.num_classes)}

        for class_folder in sorted(self.root.iterdir()):
            if class_folder.is_dir():
                if class_folder.name in NAME_TO_LABEL:
                    label = NAME_TO_LABEL[class_folder.name]
                else:
                    label = int(class_folder.name)

                if label >= self.num_classes:
                    continue

                for f in class_folder.iterdir():
                    if f.suffix == ".aedat4":
                        video_id = f.stem.split('_')[2]
                        video_per_label[label].append(video_id)

        # Podział na train/val/test
        self.train_video_ids: dict[int, set[str]] = {}
        self.val_video_ids: dict[int, set[str]] = {}
        self.test_video_ids: dict[int, set[str]] = {}

        for label, vids in video_per_label.items():
            train_paths, val_paths = train_test_split(list(set(vids)), test_size=self.val_split + self.test_split)
            val_paths, test_paths = train_test_split(val_paths, test_size=self.test_split / (self.val_split + self.test_split))

            self.train_video_ids[label] = set(train_paths)
            self.val_video_ids[label] = set(val_paths)
            self.test_video_ids[label] = set(test_paths)


    def setup(self, stage=None):
        self.train_dataset = CIFAR10DVSBinsAeadat(
            video_ids=self.train_video_ids,
            root=self.root,
            sample_bins=self.sample_bins,
            time_step=self.time_step,
            sensor_size=self.sensor_size,
            transform=self.train_transform,
            num_classes=self.num_classes,
            pos_polarity=self.pos_polarity,
            neg_polarity=self.neg_polarity,
            max_events_per_px=self.max_events_per_px,
            join_channels=self.join_channels
        )

        self.val_dataset = CIFAR10DVSBinsAeadat(
            video_ids=self.val_video_ids,
            root=self.root,
            sample_bins=self.sample_bins,
            time_step=self.time_step,
            sensor_size=self.sensor_size,
            transform=self.val_test_transform,
            num_classes=self.num_classes,
            pos_polarity=self.pos_polarity,
            neg_polarity=self.neg_polarity,
            max_events_per_px=self.max_events_per_px,
            join_channels=self.join_channels
        )

        self.test_dataset = CIFAR10DVSBinsAeadat(
            video_ids=self.test_video_ids,
            root=self.root,
            sample_bins=self.sample_bins,
            time_step=self.time_step,
            sensor_size=self.sensor_size,
            transform=self.val_test_transform,
            num_classes=self.num_classes,
            pos_polarity=self.pos_polarity,
            neg_polarity=self.neg_polarity,
            max_events_per_px=self.max_events_per_px,
            join_channels=self.join_channels
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def main():
    dm = CIFAR10DVSAedatDataModule(
        batch_size=12,
        num_workers=4,
        time_step=50_000,
        sample_bins=4,
        num_classes=10,
        pos_polarity=True,
        neg_polarity=True,
        max_events_per_px=4
    )
    dm.setup()

    print("Train dataset length:", len(dm.train_dataset))
    print("Val dataset length:", len(dm.val_dataset))
    print("Test dataset length:", len(dm.test_dataset))

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    batch_data, batch_labels = batch
    print("Batch shape - data:", batch_data.shape)
    print("Batch shape - labels:", batch_labels.shape)
    # print(batch_labels)

    # single_frame, label = batch_data[11,1,0], batch_labels[0]


if __name__ == "__main__":
    main()

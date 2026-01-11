from pathlib import Path
import tonic
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os 



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

class ResizeFrames:
    def __init__(self, size=(64, 64)):
        self.size = size

    def __call__(self, x):
        # x: (T, C, H, W)
        return torch.nn.functional.interpolate(
            x, size=self.size, mode="bilinear"
        )

def collect_cifar10dvs_files(
    root: Path,
    val_split=0.15,
    test_split=0.15,
    seed=42,
):
    files, labels = [], []

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        if class_dir.name not in NAME_TO_LABEL:
            continue

        label = NAME_TO_LABEL[class_dir.name]

        for f in class_dir.glob("*.aedat4"):
            files.append(f)
            labels.append(label)

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files,
        labels,
        test_size=val_split + test_split,
        stratify=labels,
        random_state=seed,
    )

    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files,
        temp_labels,
        test_size=test_split / (val_split + test_split),
        stratify=temp_labels,
        random_state=seed,
    )

    return train_files, train_labels, val_files, val_labels, test_files, test_labels


class CIFAR10DVSFrames(Dataset):
    def __init__(
        self,
        files,
        labels,
        sensor_size=(128, 128, 2),
        time_window=50_000,
        sample_bins=4,
        transform=None,
    ):
        self.files = files
        self.labels = labels
        self.sample_bins = sample_bins
        self.transform = transform

        self.to_frame = tonic.transforms.ToFrame(
            sensor_size=sensor_size,
            time_window=time_window,
            include_incomplete=True,
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        events = tonic.io.read_aedat4(str(self.files[idx]))
        frames = torch.from_numpy(self.to_frame(events)).float()

        T = frames.shape[0]
        if T >= self.sample_bins:
            start = torch.randint(0, T - self.sample_bins + 1, (1,)).item()
            frames = frames[start:start + self.sample_bins]
        else:
            pad = self.sample_bins - T
            frames = torch.nn.functional.pad(frames, (0, 0, 0, 0, 0, 0, 0, pad))

        if self.transform:
            frames = self.transform(frames)

        return frames, self.labels[idx]


class CIFAR10DVSDataModule(LightningDataModule):
    def __init__(
        self,
        data_root : Path = Path("data/CIFAR10DVS"),
        batch_size=8,
        num_workers=4,
        time_window=50_000,
        sample_bins=4,
        seed=42,
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.time_window = time_window
        self.sample_bins = sample_bins
        self.seed = seed

        self.transform = transforms.Compose([
            ResizeFrames((64, 64))
        ])

    def setup(self, stage=None):
        train_f, train_l, val_f, val_l, test_f, test_l = collect_cifar10dvs_files(
            self.data_root, seed=self.seed
        )

        self.train_ds = CIFAR10DVSFrames(
            train_f, train_l,
            time_window=self.time_window,
            sample_bins=self.sample_bins,
            transform=self.transform,
        )
        self.val_ds = CIFAR10DVSFrames(
            val_f, val_l,
            time_window=self.time_window,
            sample_bins=self.sample_bins,
            transform=self.transform,
        )
        self.test_ds = CIFAR10DVSFrames(
            test_f, test_l,
            time_window=self.time_window,
            sample_bins=self.sample_bins,
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    dm = CIFAR10DVSDataModule(
        batch_size=12,
        time_window=50_000,
        sample_bins=8,
    )

    dm.setup()

    x, y = next(iter(dm.train_dataloader()))
    print(x.shape)  # (B, T, C, H, W)
    print(y.shape)  # (B,)

    sample_frames = x[0]   # (T, C, H, W)
    sample_label = y[0].item()

    T, C, H, W = sample_frames.shape

    print(f"Wyświetlam {T} klatek dla próbki o etykiecie: {sample_label}")

    fig, axes = plt.subplots(1, T, figsize=(3*T, 3))
    if T == 1:
        axes = [axes]  

    for t in range(T):
        frame = sample_frames[t]  # (C, H, W)
        frame_img = frame.sum(dim=0).numpy()
        frame_img = np.transpose(frame_img)  # poprawny orient
        axes[t].imshow(frame_img, cmap="gray")
        axes[t].set_title(f"Frame {t}")
        axes[t].axis("off")
    plt.title(f"Sample label: {sample_label}")
    plt.tight_layout()
    plt.show()
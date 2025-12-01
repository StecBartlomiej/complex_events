from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path


class CIFAR10DVSBins(Dataset):
    def __init__(self, root="./data/cifar10dvs_fft_bins", transform=None):
        self.root = Path(root)
        self.transform = transform

        self.samples = []

        for class_folder in sorted(self.root.iterdir()):
            if class_folder.is_dir():
                label = int(class_folder.name)
                for f in class_folder.iterdir():
                    if f.suffix == ".npz":
                        self.samples.append((f, label))

        print(f"Loaded {len(self.samples)} precomputed bins")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)
        fft = data["fft"].astype(np.complex64)

        # (H, W) -> add channel dim for PyTorch = (1, H, W)
        fft = np.expand_dims(fft, axis=0)

        if self.transform:
            fft = self.transform(fft)
        return torch.tensor(fft), label


class CIFAR10Datamodule(LightningDataModule):
    def __init__(self, batch_size=16, num_workers=0, val_split=0.15, test_split=0.15, sensor_size=(128,128), time_step=100_000):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.sensor_size = sensor_size
        self.time_step = time_step

    def setup(self, stage=None):
        full_dataset = CIFAR10DVSBins()

        np.random.seed(42)
        indices = np.arange(len(full_dataset))
        np.random.shuffle(indices)

        n_total = len(full_dataset)
        n_test = int(n_total * self.test_split)
        n_val = int(n_total * self.val_split)
        n_train = n_total - n_val - n_test

        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:n_train+n_val].tolist()
        test_idx = indices[n_train+n_val:].tolist()

        self.train_dataset = Subset(full_dataset, train_idx)
        self.val_dataset = Subset(full_dataset, val_idx)
        self.test_dataset = Subset(full_dataset, test_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)



def main():
    dm = CIFAR10Datamodule()
    dm.setup()

    print("Train dataset length:", len(dm.train_dataset))
    print("Val dataset length:", len(dm.val_dataset))
    print("Test dataset length:", len(dm.test_dataset))

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    voxel_fft_batch, labels = batch
    print("Voxel FFT batch shape:", voxel_fft_batch.shape)
    print("Labels batch shape:", labels.shape)
    print(labels)

    
    sample_voxel_fft, label = voxel_fft_batch[0], labels[0]
    plt.imshow(np.abs(sample_voxel_fft[0].cpu().numpy()), cmap='gray')
    plt.title(f"Sample voxel FFT magnitude - label {label.item()}")
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    main()

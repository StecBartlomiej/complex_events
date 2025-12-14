from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms


def get_video_id(file_name: str):
    video_id = str(file_name).split('_')[2]
    return video_id


class CIFAR10DVSBins(Dataset):
    def __init__(self, video_ids: dict[int, set[int]], root: Path, transform=None):
        self.root = root
        self.transform = transform

        self.samples = []

        for class_folder in sorted(self.root.iterdir()):
            if class_folder.is_dir():
                label = int(class_folder.name)

                for f in class_folder.iterdir():
                    if f.suffix == ".npz":
                        video_id = get_video_id(f.name)

                        if video_id in video_ids[label]:
                            self.samples.append((f, label))

        print(f"Loaded {len(self.samples)} precomputed bins")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)
        img = data["data"]

        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR10Datamodule(LightningDataModule):
    def __init__(self, batch_size=16, num_workers=0, val_split=0.15, test_split=0.15, sensor_size=(128,128), time_step=100_000):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.sensor_size = sensor_size
        self.time_step = time_step
        self.root = Path("./data/cifar10dvs_raw_bins")
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize(mean=[0], std=[1]),
            transforms.RandomRotation(10)
            ])
        self.val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize(mean=[0], std=[1]),
            ])

    def setup(self, stage=None):
        video_per_label: dict[int, list[str]] = {}
        for i in range(10):
            video_per_label[i] = []


        # create dict with video_id per label
        for class_folder in sorted(self.root.iterdir()):
            if class_folder.is_dir():
                label = int(class_folder.name)

                for f in class_folder.iterdir():
                    if f.suffix == ".npz":
                        video_idx = get_video_id(f.name)
                        video_per_label[label].append(video_idx)

        train_video_ids: dict[int, set[int]] = {}
        val_video_ids: dict[int, set[int]] = {}
        test_video_ids: dict[int, set[int]] = {}

        for label, video_ids in video_per_label.items():
            # train: 0.7, val:0.15, test:0.15
            train_paths, val_paths = train_test_split(list(set(video_ids)), test_size=0.3)
            val_paths, test_paths = train_test_split(val_paths, test_size=0.5) 
            
            train_video_ids[label] = set(train_paths)
            val_video_ids[label] = set(val_paths)
            test_video_ids[label] = set(test_paths)

        self.train_dataset = CIFAR10DVSBins(train_video_ids, self.root, transform=self.train_transform)
        self.val_dataset = CIFAR10DVSBins(val_video_ids, self.root, transform=self.val_test_transform)
        self.test_dataset = CIFAR10DVSBins(test_video_ids, self.root, transform=self.val_test_transform)

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

class CIFAR10DVSBins3D(Dataset):
    def __init__(self, video_ids: dict[int, set[int]], root: Path, transform=None):
        self.root = root
        self.transform = transform

        self.samples = []

        for class_folder in sorted(self.root.iterdir()):
            if class_folder.is_dir():
                label = int(class_folder.name)

                for f in class_folder.iterdir():
                    if f.suffix == ".npz":
                        video_id = get_video_id(f.name)
                        if video_id in video_ids[label]:
                            self.samples.append((f, label))

        print(f"Loaded {len(self.samples)} precomputed 3D FFT windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)
        voxel_fft = data["fft"]  # teraz fft3d
        # voxel_fft = np.abs(voxel_fft)  # można użyć magnitudy FFT

        voxel_fft = torch.tensor(voxel_fft, dtype=torch.complex64).unsqueeze(0)  # dodaj channel dim

        if self.transform:
            voxel_fft = self.transform(voxel_fft)
        return voxel_fft, label


class CIFAR10Datamodule3D(LightningDataModule):
    def __init__(self, batch_size=16, num_workers=0, val_split=0.15, test_split=0.15, root="./data/cifar10dvs_fft_bins3d_test"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.root = Path(root)

    def setup(self, stage=None):
        video_per_label: dict[int, list[str]] = {i: [] for i in range(10)}

        # create dict with video_id per label
        for class_folder in sorted(self.root.iterdir()):
            if class_folder.is_dir():
                label = int(class_folder.name)
                for f in class_folder.iterdir():
                    if f.suffix == ".npz":
                        video_idx = get_video_id(f.name)
                        video_per_label[label].append(video_idx)

        train_video_ids: dict[int, set[int]] = {}
        val_video_ids: dict[int, set[int]] = {}
        test_video_ids: dict[int, set[int]] = {}

        for label, video_ids in video_per_label.items():
            train_paths, val_paths = train_test_split(list(set(video_ids)), test_size=0.3)
            val_paths, test_paths = train_test_split(val_paths, test_size=0.5)

            train_video_ids[label] = set(train_paths)
            val_video_ids[label] = set(val_paths)
            test_video_ids[label] = set(test_paths)

        self.train_dataset = CIFAR10DVSBins3D(train_video_ids, self.root)
        self.val_dataset = CIFAR10DVSBins3D(val_video_ids, self.root)
        self.test_dataset = CIFAR10DVSBins3D(test_video_ids, self.root)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)




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
    print()
    plt.imshow(np.abs(sample_voxel_fft[0].cpu().numpy()), cmap='gray')
    plt.title(f"Sample voxel FFT magnitude - label {label.item()}")
    plt.axis('off')
    plt.show()

def test_datamodule():
    dm = CIFAR10Datamodule3D(batch_size=4, num_workers=2)
    dm.setup()

    print("Train dataset length:", len(dm.train_dataset))
    print("Val dataset length:", len(dm.val_dataset))
    print("Test dataset length:", len(dm.test_dataset))

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    voxel_fft_batch, labels = batch
    print("Voxel FFT batch shape:", voxel_fft_batch.shape)  # (B, 1, num_slices, H, W)
    print("Labels batch shape:", labels.shape)

    sample_voxel_fft = voxel_fft_batch[0,0]  # pierwszy przykład, pierwszy kanał
    num_slices = sample_voxel_fft.shape[0]
    print(sample_voxel_fft[0])
    
    fig, axes = plt.subplots(1, num_slices, figsize=(num_slices*3, 3))

    for i in range(num_slices):
        axes[i].imshow(np.abs(sample_voxel_fft[i].numpy()/(sample_voxel_fft[i].numpy().max() - sample_voxel_fft[i].numpy().min()))*255, cmap='gray')
        axes[i].set_title(f"t={i}")
        axes[i].axis('off')

    plt.show()



if __name__ == "__main__":
    main()
    # test_datamodule()

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import tonic.transforms as transforms  
import torch
import tonic
import aedat
import os 
import numpy as np
import cv2
import torch.nn.functional as F


class CIFARDataset(Dataset):
    ''' CIFAR10-Dataset: n_time_bins from dataset is transformed using torch.fft2 to complex domain. 
        we have 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) 
        Each folder has 1000 recordings in aedat format. Frame size is (128, 128). 
    '''
    def __init__(self, root="./data", transform=None, sensor_size=(128,128), time_step=100_000):
        super().__init__()
        self.root = root
        self.transform = transform
        self.sensor_size = sensor_size
        self.time_step = time_step

        self.dataset_path = os.path.join(self.root, "CIFAR10DVS")

        if not os.path.exists(self.dataset_path) or not os.listdir(self.dataset_path):
            print("CIFAR10-DVS dataset not found, downloading...")
            tonic.datasets.CIFAR10DVS(save_to=self.root)

        self.classes = sorted(os.listdir(self.dataset_path))  
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(self.dataset_path, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.endswith(".aedat4"):
                    self.samples.append((os.path.join(cls_folder, fname), self.class_to_idx[cls_name]))

        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        events_data = tonic.io.read_aedat4(file_path) #format (t, x, y, p (bool))

        t = np.array([tup[0] for tup in events_data])
        x = np.array([tup[1] for tup in events_data])
        y = np.array([tup[2] for tup in events_data])
        p = np.array([tup[3].astype(np.float32) for tup in events_data])


        t_min = t.min()
        t_max = t.max()
        n_bins = int(np.ceil((t_max - t_min) / self.time_step))

        voxel = np.zeros((n_bins, self.sensor_size[0], self.sensor_size[1]), dtype=np.float32)

        bin_indices = ((t - t_min) / self.time_step).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        for b, xi, yi, pi in zip(bin_indices, x, y, p):
            voxel[b, xi, yi] += pi
            

        voxel_tensor = torch.tensor(voxel, dtype=torch.float32)

        voxel_fft = torch.fft.fft(voxel_tensor, dim=0)

        if self.transform:
            voxel_fft = self.transform(voxel_fft)
        
        return voxel_fft, label

def pad_to_max_time(voxel_tensor, max_bins):
    n_bins = voxel_tensor.shape[0]
    if n_bins < max_bins:
        pad = (0,0,0,0,0,max_bins - n_bins)  # pad tylko po osi czasowej
        voxel_tensor = F.pad(voxel_tensor, pad, "constant", 0)
    return voxel_tensor

def collate_fn(batch):
    max_bins = max([x[0].shape[0] for x in batch])
    voxels = [pad_to_max_time(x[0], max_bins) for x in batch]
    labels = [x[1] for x in batch]
    return torch.stack(voxels), torch.tensor(labels)

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
        full_dataset = CIFARDataset(root="./data",
                                    sensor_size=self.sensor_size,
                                    time_step=self.time_step)

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
                          pin_memory=True,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn)



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

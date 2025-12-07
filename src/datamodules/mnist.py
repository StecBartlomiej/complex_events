from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.transform = T.ToTensor()

    def setup(self, stage=None):
        self.train_ds = MNIST(root="data", train=True, download=True, transform=self.transform)
        self.val_ds   = MNIST(root="data", train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=2)


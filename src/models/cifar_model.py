import torch
import torch.nn.functional as F
from complextorch.nn import Linear, Conv2d, BatchNorm2d, CVCardiod, AdaptiveAvgPool2d
from pytorch_lightning import LightningModule


class ComplexCifar(LightningModule):
    def __init__(self, in_ch, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()


        # A simple complex CNN: two complex conv blocks -> complex linear classifier
        # Input: (N, 28, 128, 128) real -> converted to complex with zero imaginary part
        self.conv1 = Conv2d(in_ch, 16, kernel_size=3, padding=1)
        self.act1 = CVCardiod()
        self.pool1 = AdaptiveAvgPool2d(in_ch // 2)


        self.conv2 = Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = CVCardiod()
        self.pool2 = AdaptiveAvgPool2d(in_ch // 4)


        self.fc1 = Linear(32 * ((in_ch // 4) ** 2), 128)
        self.fc_act = CVCardiod()

        self.classifier = Linear(128, 10)


    def forward(self, x_complex):
        x = self.conv1(x_complex)
        x = self.act1(x)
        x = self.pool1(x)


        # conv block 2
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc_act(x)


        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        assert(x.dtype == torch.complex64)
        logits_complex = self(x)

        # convert complex logits to real logits by using magnitude (abs)
        logits_real = torch.abs(logits_complex)

        loss = F.cross_entropy(logits_real, y)
        preds = torch.argmax(logits_real, dim=1)
        acc = (preds == y).float().mean()


        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(dtype=torch.complex64)
        logits_complex = self(x)
        logits_real = torch.abs(logits_complex)


        loss = F.cross_entropy(logits_real, y)
        preds = torch.argmax(logits_real, dim=1)
        acc = (preds == y).float().mean()


        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(dtype=torch.complex64)
        logits_complex = self(x)
        logits_real = torch.abs(logits_complex)


        loss = F.cross_entropy(logits_real, y)
        preds = torch.argmax(logits_real, dim=1)
        acc = (preds == y).float().mean()


        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr) # type: ignore
        return opt


from complextorch import nn as cnn
import torch
import torch.nn.functional as F
from torch import nn
from complextorch.nn import Linear, Conv2d, CVCardiod, AdaptiveAvgPool2d
from pytorch_lightning import LightningModule
import complextorch.nn.functional as cvF
import torchcvnn.nn
import utils.complex_layers as cl




class ComplexCifar(LightningModule):
    def __init__(self, in_ch, img_size=64, lr: float = 1e-3):
        super().__init__()
        self.in_ch = in_ch
        self.img_size = img_size
        self.save_hyperparameters()

        self.act = cnn.CVCardiod()

        self.conv1 = cl.FrequencyConv2D(self.in_ch, 16, kernel_size=64)
        self.norm1 = cl.FrequencyInstanceNorm2D(16)
        self.pool1 = cl.ComplexAdaptiveAvgPool2d(32)

        self.conv2 = cl.FrequencyConv2D(16, 24, kernel_size=32)
        self.norm2 = cl.FrequencyInstanceNorm2D(24)
        self.pool2 = cl.ComplexAdaptiveAvgPool2d(16)

        self.conv3 = cl.FrequencyConv2D(24, 32, kernel_size=16)
        self.norm3 = cl.FrequencyInstanceNorm2D(32)
        self.pool3 = cl.ComplexAdaptiveAvgPool2d(8)
        
        # self.conv4 = cl.FrequencyConv2D(128, 256, kernel_size=8)
        # self.norm4 = cl.FrequencyInstanceNorm2D(256)
        # self.pool4 = cl.ComplexAdaptiveAvgPool2d(4)

        self.dropout = cl.ComplexDropout(p=0.3)
        self.fc1 = Linear(32 * 8 * 8, 512)
        self.fc2 = Linear(512, 256)
        self.classifier = Linear(256, 10)

    def forward(self, input):
        x_complex = torch.fft.fft2(input)

        x = self.conv1(x_complex)
        x = self.norm1(x)
        x = self.act(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)
        x = self.pool3(x)

        # x = self.conv4(x)
        # x = self.norm4(x)
        # x = self.act(x)
        # x = self.pool4(x)

        x = torch.flatten(x, 1)

        if self.training:
            x = self.dropout(x) 

        x = self.fc1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.act(x)

        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
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
        # x = x.to(dtype=torch.complex64)
        logits_complex = self(x)
        logits_real = torch.abs(logits_complex)


        loss = F.cross_entropy(logits_real, y)
        preds = torch.argmax(logits_real, dim=1)

        acc = (preds == y).float().mean()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        # x = x.to(dtype=torch.complex64)
        logits_complex = self(x)
        logits_real = torch.abs(logits_complex)


        loss = F.cross_entropy(logits_real, y)
        preds = torch.argmax(logits_real, dim=1)

        acc = (preds == y).float().mean()

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)


    def configure_optimizers(self): # type: ignore
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr) # type: ignore
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        return {
                'optimizer': opt,
                'lr_scheduler': scheduler,
                'monitor': 'train_loss'
                }


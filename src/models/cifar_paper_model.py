import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning import LightningModule
from utils.complex_layers import *



class ComplexCifarPaper(LightningModule):
    def __init__(self, in_ch, lr: float = 1e-3):
        super(ComplexCifarPaper, self).__init__()
        self.in_ch = in_ch
        self.save_hyperparameters()

        self.conv1 = FrequencyConv2D(in_ch, 32, kernel_size=64)
        self.ln1 = FrequencyInstanceNorm2D(32)

        self.layer1 = self._make_layer(32, 64, kernel_size=32, num_blocks=1)
        self.layer2 = self._make_layer(64, 128, kernel_size=16, num_blocks=1)
        self.layer3 = self._make_layer(128, 256, kernel_size=8, num_blocks=1)
        # self.layer4 = self._make_layer(512, 512, kernel_size=4, num_blocks=1)

        self.pooling_layer = ComplexAdaptiveAvgPool2d(output_size=(32, 32))
        self.pooling_layer1 = ComplexAdaptiveAvgPool2d(output_size=(16, 16))
        self.pooling_layer2 = ComplexAdaptiveAvgPool2d(output_size=(8, 8))
        # self.pooling_layer3 = ComplexAdaptiveAvgPool2d(output_size=(4, 4))


        self.avgpool = ComplexAdaptiveMaxPool2d(output_size=(1, 1))
        self.fc2 = FrequencyLinear(256, 10)
        self.dropout = ComplexDropout(p=0.3)

    def _make_layer(self, in_channels, out_channels, kernel_size, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, input):
        x_complex = torch.fft.fft2(input)

        x = LOG_Magnitude(self.ln1(self.conv1(x_complex)))
        x = self.pooling_layer(x)


        x = self.layer1(x)
        x = self.pooling_layer1(x)


        x = self.layer2(x)
        x = self.pooling_layer2(x)


        x = self.layer3(x)
        # x = self.pooling_layer3(x)


        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc2(x)

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


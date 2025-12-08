from complextorch import nn as cnn
import torch
import torch.nn.functional as F
from torch import nn
from complextorch.nn import Linear, Conv2d, CVCardiod, AdaptiveAvgPool2d
from pytorch_lightning import LightningModule
import complextorch.nn.functional as cvF
import torchcvnn.nn


# Try maxpool on amplitude of complex number -> preserves phase
class CVMaxPool2D(torch.nn.MaxPool2d):
    def __init__(self, kernel_size=(2, 2)) -> None:
        super().__init__(kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cvF.apply_complex_split(super().forward, super().forward, input)


class AbsMaxPool2D(nn.Module):
    def __init__(self, kernel_size=2): 
        super().__init__()
        self.kernel_size=kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mag = input.abs()

        mag_max, max_idx = nn.functional.max_pool2d(mag,
                                              kernel_size=self.kernel_size,
                                              return_indices=True)
        x = torch.flatten(input, 2)
        output = torch.gather(x, 2, torch.flatten(max_idx, 2)).view(mag_max.size())
        return output



class FrequencyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FrequencyConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if in_channels != out_channels:
            self.filters = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.complex64))
        else:
            self.filters = nn.Parameter(torch.randn(out_channels, kernel_size, kernel_size, dtype=torch.complex64))
        self.Initialization()

    def Initialization(self):
        bound = 1 / (self.in_channels ** 0.5)  
        nn.init.uniform_(self.filters.real, -bound, bound)
        nn.init.uniform_(self.filters.imag, -bound, bound)

    def forward(self, inputs):
        batch_size, in_channels, height, width = inputs.shape
        if self.in_channels != self.out_channels:
            inputs_expanded = inputs.unsqueeze(1)
            filters_expanded = self.filters.unsqueeze(0)
            freq_output = inputs_expanded * filters_expanded
            freq_output = freq_output.sum(dim=2)
        else:
            freq_output = inputs * self.filters
        return freq_output

# TODO: add instance normalization
class ResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel):
        super().__init__()
        self.act = cnn.CVCardiod()

        self.conv1 = FrequencyConv2D(ch_in, ch_out, kernel_size=kernel)
        self.conv2 = FrequencyConv2D(ch_out, ch_out, kernel_size=kernel)

        self.norm = torchcvnn.nn.BatchNorm2d(ch_out)
        
        self.residual = None
        if ch_in != ch_out:
            self.residual = FrequencyConv2D(ch_in, ch_out, kernel_size=kernel)
            
    def forward(self, x):
        skip = x if self.residual is None else self.residual(x) 

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        x = x + skip

        x = self.act(x)
        x = self.norm(x)

        return x


class ComplexCifar(LightningModule):
    def __init__(self, in_ch, lr: float = 1e-3):
        super().__init__()
        self.in_size = 128
        self.save_hyperparameters()

        # A simple complex CNN: two complex conv blocks -> complex linear classifier
        # Input: (N, 1, 128, 128) real -> converted to complex with zero imaginary part

        self.act = cnn.CVCardiod()

        self.conv1 = FrequencyConv2D(in_ch, 32, kernel_size=64)
        self.norm1 = torchcvnn.nn.BatchNorm2d(32)
        self.pool1 = AdaptiveAvgPool2d(32)

        self.conv2 = FrequencyConv2D(32, 64, kernel_size=32)
        self.norm2 = torchcvnn.nn.BatchNorm2d(64)
        self.pool2 = AdaptiveAvgPool2d(16)

        self.conv3 = FrequencyConv2D(64, 128, kernel_size=16)
        self.norm3 = torchcvnn.nn.BatchNorm2d(128)
        self.pool3 = AdaptiveAvgPool2d(8)
        
        self.conv4 = FrequencyConv2D(128, 256, kernel_size=8)
        self.norm4 = torchcvnn.nn.BatchNorm2d(256)
        self.pool4 = AdaptiveAvgPool2d(4)

        self.fc1 = Linear(256 * 4 * 4, 256)
        self.fc2 = Linear(256, 128)
        self.classifier = Linear(128, 10)

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

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.act(x)
        x = self.pool4(x)

        x = torch.flatten(x, 1)

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


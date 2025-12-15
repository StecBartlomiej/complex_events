from complextorch import nn as cnn
import torch
import torch.nn.functional as F
from torch import nn
from complextorch.nn import Linear, Conv2d, CVCardiod, AdaptiveAvgPool2d
from pytorch_lightning import LightningModule
import complextorch.nn.functional as cvF
import torchcvnn.nn

ACTIVATION_OPTION = 0

## ========================= FUNKCJE POMOCNICZE ========================= ##

def complex_to_real_imag(complex_tensor):
    real_part = complex_tensor.real
    imag_part = complex_tensor.imag
    return real_part,imag_part,torch.cat((real_part, imag_part), dim=1)

def LOG_Magnitude(x_complex):
    mag = torch.abs(x_complex)
    mag_transformed = torch.log1p(mag)
    phase = torch.angle(x_complex)
    output = mag_transformed * torch.exp(1j * phase)
    return output

def act_funtion(x, option = 0):
    if option == 0:
        return LOG_Magnitude(x)
    elif option == 1:
        return cnn.CVCardiod()(x)
    
## ========================= WARSTWY ZESPOLONE ========================= ##

class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        real = self.dropout(x.real)
        imag = self.dropout(x.imag)
        return torch.complex(real, imag)
    
class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super(ComplexAdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size
        self.pool_real = nn.AdaptiveAvgPool2d(output_size)
        self.pool_imag = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, inputs):
        real_part, imag_part, _ = complex_to_real_imag(inputs)
        pooled_real = self.pool_real(real_part)
        pooled_imag = self.pool_imag(imag_part)
        pooled_output = torch.complex(pooled_real, pooled_imag)
        return pooled_output   
    
class ComplexAdaptiveMaxPool2d(nn.Module):
    def __init__(self, output_size):
        super(ComplexAdaptiveMaxPool2d, self).__init__()
        self.output_size = output_size
        self.pool_real = nn.AdaptiveMaxPool2d(output_size)
        self.pool_imag = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        real = x.real
        imag = x.imag
        pooled_real = self.pool_real(real)
        pooled_imag = self.pool_imag(imag)
        pooled = torch.complex(pooled_real, pooled_imag)
        return pooled


class FrequencyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(FrequencyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.filters = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.complex64))
        self.Initialization()

    def Initialization(self):
        bound = 1 / (self.in_features ** 0.5)
        nn.init.uniform_(self.filters.real, -bound, bound)
        nn.init.uniform_(self.filters.imag, -bound, bound)
    def forward(self, inputs):
        freq_output = inputs.unsqueeze(1) * self.filters 
        freq_output = freq_output.sum(-1)
        return freq_output

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


class FrequencyInstanceNorm2D(nn.Module):
    def __init__(self, num_features):
        super(FrequencyInstanceNorm2D, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features, dtype=torch.complex64))
        self.beta = nn.Parameter(torch.zeros(num_features, dtype=torch.complex64))

    def forward(self, inputs):
        mean = torch.mean(inputs, dim=(-2, -1), keepdim=True)
        var = torch.var(inputs, dim=(-2, -1), unbiased=False, keepdim=True)
        freq_normalized = (inputs - mean) / torch.sqrt(var + 1e-5)
        freq_output = self.gamma.view(1, -1, 1, 1) * freq_normalized + self.beta.view(1, -1, 1, 1)
        return freq_output


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
    

class ResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel):
        super().__init__()
        self.conv1 = FrequencyConv2D(ch_in, ch_out, kernel)
        self.ln1 = FrequencyInstanceNorm2D(ch_out)
        self.conv2 = FrequencyConv2D(ch_out, ch_out, kernel)
        self.ln2 = FrequencyInstanceNorm2D(ch_out)


        self.skip_connection = nn.Sequential()
        if ch_in != ch_out:
            self.skip_connection = nn.Sequential(
                FrequencyConv2D(ch_in, ch_out, kernel_size = 1),
                FrequencyInstanceNorm2D(ch_out)
            )

            
    def forward(self, x):

        out = self.ln1(self.conv1(x))
        out = LOG_Magnitude(out)        #tutaj można dodać Cartoide na przykład, paper mówi żeby dać taką
        out = self.ln2(self.conv2(out)) 
        out = out + self.skip_connection(x)
        out = LOG_Magnitude(out)

        return out
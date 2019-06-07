import torch
import torch.nn as nn
import torch.nn.functional as F

import args

class MaskedConv2d(nn.Conv2d):
    """
    This is a convolution module modified to satisfy the
    causality (future pixels are not allowed to influence the
    current pixel).
    We don't support masking over the channels, since the example is
    only on MNIST. 
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 mask_type: str = 'B', 
                 stride: int = 1):
        """
        Args:
          in_channels:  Number of input channels.
          out_channels: Number of output channels.
          kernel_size:  Size of conv kernel.
          mask_type:    Mask type 'A' for first layer and 'B' elsewhere
                        This is used so the center pixel in the first layer
                        does not "see" the input from future.
          stride:       Controls the stride.
        """
        super().__init__(
            in_channels, out_channels, kernel_size, 
            stride, padding=kernel_size//2
            )

        assert mask_type in ('A', 'B'), "Mask type must be 'A' or 'B'"

        mask = torch.ones_like(self.weight.data)
        mask[:, :, kernel_size//2, kernel_size//2 + (mask_type == 'B'):] = 0
        mask[:, :, kernel_size//2+1:] = 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class GatedResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_classes: int = 10,
                 kernel_size: int = 3,
                 stride: int = 1,
                 aux_channels: int = 0):
        """
        Args:
          in_channels:  Number of input channels.
          out_channels: Number of output channels.
          n_classes:    Number of output classes in the data
          kernel_size:  Size of conv kernel.
          stride:       Controls the stride.
          aux_channels: Number of additional channels in conv blocks
        """
        super().__init__()

        self.out_channels = out_channels
        # resblock is only after the first conv layer, so we use mask 'B'
        self.conv = MaskedConv2d(
            in_channels, out_channels*2, kernel_size, 'B', stride
            )
        self.y_embed = nn.Linear(n_classes, out_channels*2)

        if aux_channels != 0 and aux_channels != out_channels*2:
            self.aux_shortcut = nn.Sequential(
                nn.Conv2d(aux_channels, out_channels*2, 1),
                nn.BatchNorm2d(out_channels*2, momentum=0.1)
            )
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.1)
            )
        
        self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, x, y):
        if x.dim() == 5:
            x, aux = torch.split(x, 1, dim=0)
            x = torch.squeeze(x, 0)
            aux = torch.squeeze(x, 0)
        else:
            aux = None
        x1 = self.conv(x)
        y = torch.unsqueeze(torch.unsqueeze(self.y_embed(y), -1), -1)
        if aux is not None:
            if hasattr(self, 'aux_shortcut'):
                aux = self.aux_shortcut(aux)                
            x1 = (x1 + aux)/2
        # split for gate (note: pytorch dims are [n, c, h, w])
        xf, xg = torch.split(x1, self.out_channels, dim=1)
        yf, yg = torch.split(y, self.out_channels, dim=1)
        f = torch.tanh(xf + yf)
        g = torch.sigmoid(xg + yg)
        if hasattr(self, 'shortcut'):
            x = self.shortcut(x)
            
        return x + self.batchnorm(g * f)
    
class PixelCNN(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 n_classes: int,
                 n_features: int,
                 n_layers: int,
                 n_bins: int,
                 dropout:float = 0.5):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        
        self.input_batchnorm = nn.BatchNorm2d(in_channels, momentum=0.1)
        for l in range(n_layers):
            if l == 0:
                block = nn.Sequential(
                    MaskedConv2d(in_channels=in_channels+1,
                                 out_channels=n_features,
                                 kernel_size=7,
                                 mask_type='A'),
                    nn.BatchNorm2d(n_features, momentum=0.1),
                    nn.ReLU()
                )
            else:
                block = GatedResBlock(n_features, n_features, n_classes)
            self.layers.append(block)
        
        # Down Pass
        for _ in range(n_layers):
            block = GatedResBlock(
                n_features, 
                n_features, 
                n_classes, 
                aux_channels=n_features
            )
            self.layers.append(block)
            
        # Last layer: project to n_bins (output is [-1, n_bins, h, w])
        self.dropout = nn.Dropout2d(dropout)
        self.layers.append(GatedResBlock(n_features, n_bins, n_classes))
        self.layers.append(nn.LogSoftmax(dim=1))
        
    def forward(self, x, y):
        x = F.pad(x, (0, 0, 0, 0, 0, 1, 0, 0), mode='constant', value=1)
        
        # up pass
        features = []
        i = -1
        for _ in range(self.n_layers):
            i += 1
            if i > 0:
                x = self.layers[i](x, y)
            else:
                x = self.layers[i](x)
            features.append(x)
        
        for _ in range(self.n_layers):
            i += 1
            x = self.layers[i](torch.stack((x, features.pop())), y)
            
        # Last layer
        x = self.dropout(x)
        i += 1
        x = self.layers[i](x, y)
        i += 1
        x = self.layers[i](x)
        
        # some sanity checks
        assert i == len(self.layers) - 1
        assert len(features) == 0
        
        return x
        
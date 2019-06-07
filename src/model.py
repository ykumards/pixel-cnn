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
          n_classes:    Number of output classes in the data.
          kernel_size:  Size of conv kernel.
          stride:       Controls the conv stride.
          aux_channels: Number of additional channels in conv blocks
        """
        super().__init__()

        self.out_channels = out_channels
        # - resblock is only after the first conv layer, so we use mask 'B'
        # - use 2*out_channels coz we're splitting them for the gate
        self.conv = MaskedConv2d(
            in_channels, out_channels*2, kernel_size, 'B', stride
            )
        self.y_embed = nn.Linear(n_classes, out_channels*2)

        if aux_channels != 0 and aux_channels != out_channels*2:
            self.aux_shortcut = nn.Sequential(
                nn.Conv2d(aux_channels, out_channels*2, 1),
                nn.BatchNorm2d(out_channels*2, momentum=0.1)
            )
        
        # for last block
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), #1x1 convolution
                nn.BatchNorm2d(out_channels, momentum=0.1)
            )
        
        self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, x, y):
        if x.dim() == 5: # normally we have [bs, ch, h, w]
            # separate out aux and x
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
                 n_inter_channels: int,
                 n_layers: int,
                 n_output_bins: int,
                 dropout:float = 0.5):
        """
        Args:
          in_channels:      Number of input channels.
          n_classes:        Number of classes in the dataset.
          n_inter_channels: Number of channels in intermediate layers.
          n_layers:         Number of layers in the model 
                            (including mask A convolution).
          n_output_bins:    Number of output bins we get after discretizing
                            the pixel values in the input images. 
                            Eg: defaults to 4 => 
                            0-0.25=0; 0.25-0.5=1; 0.5-0.75=2; 0.75-1=3
          dropout:          Dropout probability, defaults to 0.5
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.input_batchnorm = nn.BatchNorm2d(in_channels, momentum=0.1)
        for l in range(n_layers):
            if l == 0:
                block = nn.Sequential(
                    MaskedConv2d(in_channels=in_channels+1,
                                 out_channels=n_inter_channels,
                                 kernel_size=7,
                                 mask_type='A'), # first conv is type A
                    nn.BatchNorm2d(n_inter_channels, momentum=0.1),
                    nn.ReLU()
                )
            else:
                block = GatedResBlock(in_channels=n_inter_channels, 
                                      out_channels=n_inter_channels, 
                                      n_classes=n_classes)
            self.layers.append(block)
        
        # Down Pass
        for _ in range(n_layers):
            block = GatedResBlock(
                in_channels=n_inter_channels, 
                out_channels=n_inter_channels, 
                n_classes=n_classes, 
                aux_channels=n_inter_channels
            )
            self.layers.append(block)
            
        # Last layer: project to n_output_bins (output is [-1, n_output_bins, h, w])
        self.dropout = nn.Dropout2d(dropout)
        self.layers.append(GatedResBlock(
            in_channels=n_inter_channels, out_channels=n_output_bins, n_classes=n_classes))
        self.layers.append(nn.LogSoftmax(dim=1))
        
    def forward(self, x, y):
        # Add channel of ones so network can tell where padding is
        x = F.pad(x, (0, 0, 0, 0, 0, 1, 0, 0), mode='constant', value=1)
        
        # skip connections as similar to pixelcnn++ paper because it is
        # much faster to train and we obtain good results without tuning for ages
        skip_conn_stack = []
        layer_idx = -1
        for _ in range(self.n_layers):
            layer_idx += 1
            if layer_idx > 0:
                x = self.layers[layer_idx](x, y)
            else:
                # first layer (A masked conv) only takes x in forward pass
                x = self.layers[layer_idx](x)
            skip_conn_stack.append(x)
            
        for _ in range(self.n_layers):
            layer_idx += 1
            skip_conn = skip_conn_stack.pop()
            x = self.layers[layer_idx](torch.stack((x, skip_conn)), y)
            
        # Last layer
        x = self.dropout(x)
        layer_idx += 1
        x = self.layers[layer_idx](x, y)
        layer_idx += 1
        x = self.layers[layer_idx](x)
        
        # some sanity checks
        assert layer_idx == len(self.layers) - 1
        assert len(skip_conn_stack) == 0, "skip_conn stack must be empty"
        
        return x
        
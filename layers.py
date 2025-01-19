import torch
import torch.nn as nn

class BasicResidualBlock(nn.Module):
    '''
    A basic residual block for ResNet architectures, featuring two convolutional layers and a downsample path 
    for skip connections.

    Parameters:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor after the block.
        stride (int, optional): Stride for the first convolutional layer; used for downsampling. Default is 1.
        Conv (class, optional): Convolutional layer type, e.g., `nn.Conv1d` for 1D data. Default is `nn.Conv1d`.
        BatchNorm (class, optional): Normalization layer type, e.g., `nn.BatchNorm1d` for 1D data. Default is `nn.BatchNorm1d`.
        Activ (class, optional): Activation function to use. Default is `nn.ReLU`.

    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, Conv = nn.Conv1d, BatchNorm = nn.BatchNorm1d, Activ = nn.ReLU):
        super(BasicResidualBlock, self).__init__()

        # First Convolutional layer
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn1 = BatchNorm(out_channels)

        # Second Convolutional layer
        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm(out_channels)

        # Define downsample layer to skip connection if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BatchNorm(out_channels)
            )

        self.activ = Activ()

    def forward(self, z):

        # Skip connection (identity function)
        id = z
        # Apply downsampling if needed
        if self.downsample is not None:
            id = self.downsample(id)

        # Main path
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.activ(z)

        z = self.conv2(z)
        z = self.bn2(z)

        # Add skip connection
        z += id
        z = self.activ(z)

        return z



class CoordConv1d(nn.Module):
    '''
    CoordConv was initially proposed to add coordinate information to convolutional layers, 
    allowing the network to learn spatial relationships more explicitly. Coordinates
    are concatenated to the inputs in each forward pass.
    See: https://doi.org/10.48550/arXiv.1807.03247

    Discussion:
    - Allowing one explicitly to pass the coordinates in the forward() method can be useful when the coordinate partition is not uniformly spaced
    - However, in models where channels are changing might not be useful.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CoordConv1d, self).__init__()
        
        # Define a regular 1D convolution with an extra input channel for coordinates
        self.conv = nn.Conv1d(in_channels + 1, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        # Placeholder for the coordinate tensor (will be created lazily)
        self.coords = None

    def forward(self, z):
        # z has shape (N, C, D), where N is batch size, C is channels, and D is data length
        N, C, D = z.size()
                
        if self.coords is None or self.coords.size(-1) != D:
            self.coords = torch.linspace(0, 1, D, device=z.device).view(1, 1, D)
            self.coords.requires_grad = False  # This tensor should not require gradients
        
        # Expand the coords tensor to match the batch size dynamically
        coords = self.coords.expand(N, 1, D)  # Shape (N, C, D)
        
        # Concatenate the coordinate channel with the input along the channel dimension
        z = torch.cat([z, coords], dim=1)  # New shape: (N, C + 1, D)
        
        # Apply the convolution
        z = self.conv(z)
        
        return z
    
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{self.conv.in_channels - 1}, {self.conv.out_channels}, '
                f'kernel_size={self.conv.kernel_size}, stride={self.conv.stride}, '
                f'padding={self.conv.padding}, dilation={self.conv.dilation}, '
                f'groups={self.conv.groups}, bias={self.conv.bias is not None})')
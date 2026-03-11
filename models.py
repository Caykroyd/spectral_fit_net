import torch
import torch.nn as nn
import torch.nn.functional as F

from math import ceil

from layers import BasicResidualBlock, CoordConv1d, CoordConv1d_2


# class GaussianInferenceModel(nn.Module):
#     '''
#     Description: The models perform a simultaneous k-Gaussian fit across multiple
#     spectral lines. Each Gaussian is tied together by wavelength constraints
#     across lines.
#     The model infers the parameters of each Gaussian (amplitude, mean, and
#     standard deviation) from an input spectrum.

#     Input data:
#       Spectra containing tied spectral lines.
#       shape (N, C, D) = (batch_size, spectal line count, spectral line data length)

#     Output data:
#       Inferred parameters for each Gaussian.
#       shape (N, O) = (batch_size, out_channels)
#     '''
#     def __init__(self, in_channels, num_gaussians, signal_length):
#         super(FirstGaussNet, self).__init__()


# Remarks: Beware of pooling operations as they reduce the spatial resolution of
# the spectrum. Consequently, subsequent layer’s output will have progressively
# coarser information about the location of the features, which can be an issue
# for tasks where accurate positional information is required (i.e. regression).


class FirstGaussNet(nn.Module):
    '''
    First network architecture to try. It works pretty good, despite the pooling.
    '''
    def __init__(self, in_channels, out_channels, signal_length):
        super(FirstGaussNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * (signal_length // 8), 128) # input size is halved with each pooling
        self.fc2 = nn.Linear(128, out_channels) # Output: (amplitude, mean, std) = 3 parameters per gaussian
        self.dropout = nn.Dropout(p=0.2) # help avoid overfitting
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.activ(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.activ(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.activ(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)  # flatten the last dims
        
        x = self.fc1(x)
        x = self.activ(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x
    
class SimpleGaussNet(nn.Module):
    '''
    One signal-wise global pooling at the end.
    '''
    def __init__(self, in_channels, out_channels, signal_length):

        super(SimpleGaussNet, self).__init__()

        # Initial 1D convolution layer
        self.conv1 = nn.Conv1d(in_channels, 8*in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(8*in_channels)
        self.conv2 = nn.Conv1d(8*in_channels, 16*in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(16*in_channels)
        self.conv3 = nn.Conv1d(16*in_channels, 32*in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(32*in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool1d(kernel_size = signal_length, return_indices=True)

        self.fc1 = nn.Linear(2*32*in_channels, 8*out_channels)
        self.fc2 = nn.Linear(8*out_channels, out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, z):
        signal_length = z.size(-1)

        z = self.conv1(z)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.conv2(z)
        z = self.bn2(z)
        z = self.relu(z)

        z = self.conv3(z)
        z = self.bn3(z)
        z = self.relu(z)
        
        # z = self.conv4(z)
        # z = self.bn4(z)
        # z = self.relu(z)

        z, idx = self.pool(z) # idx returned by MaxPool2d will contain the flattened indices of the matrix (C, D).
        idx = idx % signal_length # We require indices only along D dimension.
        x = idx / signal_length # normalise to [0, 1]

        z = torch.stack([z, x], dim=-1)
        z = torch.flatten(z, 1)  # (N, 2*32*C)

        z = self.fc1(z)  
        z = self.dropout(z)
        z = self.relu(z)

        z = self.fc2(z)  

        return z # Output shape: (batch_size, out_channels)
    
    
class FlatGaussNet(nn.Module):
    '''
    Simple architecture with no pooling so that positional information does not get lost.
    Three convolution layers work best.
    '''
    def __init__(self, in_channels, out_channels, signal_length):

        super(FlatGaussNet, self).__init__()

        # Initial 1D convolution layer
        self.conv1 = nn.Conv1d(in_channels, 4*in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(4*in_channels)
        self.conv2 = nn.Conv1d(4*in_channels, 8*in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(8*in_channels)
        self.conv3 = nn.Conv1d(8*in_channels, 16*in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(16*in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(16*in_channels*signal_length, 8*out_channels)
        self.fc2 = nn.Linear(8*out_channels, out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, z):
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.conv2(z)
        z = self.bn2(z)
        z = self.relu(z)

        z = self.conv3(z)
        z = self.bn3(z)
        z = self.relu(z)
        
        z = torch.flatten(z, 1) 

        z = self.fc1(z)  
        z = self.dropout(z)
        z = self.relu(z)

        z = self.fc2(z)  

        return z # Output shape: (batch_size, out_channels)
    
    
class CoordGaussNet(nn.Module):
    '''
    Network architecture using Coordinate Convolution layers.
    '''
    def __init__(self, in_channels, out_channels, signal_length):
        super(CoordGaussNet, self).__init__()
        self.conv1a = CoordConv1d(in_channels, 8*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.bn1a = nn.BatchNorm1d(8*in_channels)
        self.conv1b = CoordConv1d(8*in_channels, 16*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.bn1b = nn.BatchNorm1d(16*in_channels)
        self.pool1 = nn.MaxPool1d(2, stride=2)

        self.conv2a = CoordConv1d(16*in_channels, 32*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.bn2a = nn.BatchNorm1d(32*in_channels)
        self.conv2b = CoordConv1d(32*in_channels, 32*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.bn2b = nn.BatchNorm1d(32*in_channels)
        self.pool2 = nn.MaxPool1d(2, stride=2)

        self.conv3a = CoordConv1d(32*in_channels, 64*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.bn3a = nn.BatchNorm1d(64*in_channels)
        self.conv3b = CoordConv1d(64*in_channels, 64*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.bn3b = nn.BatchNorm1d(64*in_channels)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        
        self.conv4a = CoordConv1d(64*in_channels, 64*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.bn4a = nn.BatchNorm1d(64*in_channels)
        self.conv4b = CoordConv1d(64*in_channels, 64*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.bn4b = nn.BatchNorm1d(64*in_channels)
        self.pool4 = nn.MaxPool1d(2, stride=2)
        self.activ = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(64*in_channels*(signal_length//16), 32*out_channels)
        self.fc2 = nn.Linear(32*out_channels, out_channels)

        self.dropout = nn.Dropout(p=0.3) # help avoid overfitting

    def forward(self, z):
        z = self.conv1a(z)
        z = self.activ(z)
        z = self.bn1a(z)
        z = self.conv1b(z)
        z = self.activ(z)
        z = self.bn1b(z)
        z = self.pool1(z)

        z = self.conv2a(z)
        z = self.activ(z)
        z = self.bn2a(z)
        z = self.conv2b(z)
        z = self.activ(z)
        z = self.bn2b(z)
        z = self.pool2(z)

        z = self.conv3a(z)
        z = self.activ(z)
        z = self.bn3a(z)
        z = self.conv3b(z)
        z = self.activ(z)
        z = self.bn3b(z)
        z = self.pool3(z)
        
        z = self.conv4a(z)
        z = self.activ(z)
        z = self.bn4a(z)
        z = self.conv4b(z)
        z = self.activ(z)
        z = self.bn4b(z)
        z = self.pool4(z)

        z = torch.flatten(z, 1) 

        z = self.fc1(z)  
        z = self.dropout(z)
        z = self.sigmoid(z)

        z = self.fc2(z)  
        # z = self.sigmoid(z) # Constraint output to [0, 1]

        return z # Output shape: (batch_size, (2 * in_channels + 1) * num_gaussians)
    

class CoordGaussNet_2(nn.Module):
    '''
    Network architecture using Coordinate Convolution layers.
    '''
    def __init__(self, in_channels, out_channels, signal_length, groups=1):
        super(CoordGaussNet_2, self).__init__()
        self.conv1a = CoordConv1d_2(in_channels, 8*in_channels, kernel_size=5, stride=1, padding=1, bias=False, groups=groups)
        self.bn1a = nn.BatchNorm1d(8*in_channels)
        self.conv1b = CoordConv1d_2(8*in_channels, 16*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn1b = nn.BatchNorm1d(16*in_channels)
        self.pool1 = nn.MaxPool1d(2, stride=2)

        self.conv2a = CoordConv1d_2(16*in_channels, 32*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn2a = nn.BatchNorm1d(32*in_channels)
        self.conv2b = CoordConv1d_2(32*in_channels, 32*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn2b = nn.BatchNorm1d(32*in_channels)
        self.pool2 = nn.MaxPool1d(2, stride=2)

        self.conv3a = CoordConv1d_2(32*in_channels, 64*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn3a = nn.BatchNorm1d(64*in_channels)
        self.conv3b = CoordConv1d_2(64*in_channels, 64*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn3b = nn.BatchNorm1d(64*in_channels)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        
        self.conv4a = CoordConv1d_2(64*in_channels, 128*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn4a = nn.BatchNorm1d(128*in_channels)
        self.conv4b = CoordConv1d_2(128*in_channels, 128*in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn4b = nn.BatchNorm1d(128*in_channels)
        self.pool4 = nn.MaxPool1d(2, stride=2)
        self.activ = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(128*in_channels*(signal_length//16), 32*out_channels)
        self.fc2 = nn.Linear(32*out_channels, out_channels)

        self.dropout = nn.Dropout(p=0.3) # help avoid overfitting

    def forward(self, z):
        z = self.conv1a(z)
        z = self.activ(z)
        z = self.bn1a(z)
        z = self.conv1b(z)
        z = self.activ(z)
        z = self.bn1b(z)
        z = self.pool1(z)

        z = self.conv2a(z)
        z = self.activ(z)
        z = self.bn2a(z)
        z = self.conv2b(z)
        z = self.activ(z)
        z = self.bn2b(z)
        z = self.pool2(z)

        z = self.conv3a(z)
        z = self.activ(z)
        z = self.bn3a(z)
        z = self.conv3b(z)
        z = self.activ(z)
        z = self.bn3b(z)
        z = self.pool3(z)
        
        z = self.conv4a(z)
        z = self.activ(z)
        z = self.bn4a(z)
        z = self.conv4b(z)
        z = self.activ(z)
        z = self.bn4b(z)
        z = self.pool4(z)

        z = torch.flatten(z, 1) 

        z = self.fc1(z)  
        z = self.dropout(z)
        z = self.sigmoid(z)

        z = self.fc2(z)  
        # z = self.sigmoid(z) # Constraint output to [0, 1]

        return z # Output shape: (batch_size, (2 * in_channels + 1) * num_gaussians)
    

    
class ResCoordGaussNet(nn.Module):
    '''
    Network architecture using Residual Blocks with Coordinate Convolution layers.
    '''
    def __init__(self, in_channels, out_channels, signal_length):
        super(ResCoordGaussNet, self).__init__()

        
        # Initial 1D convolution layer
        self.conv1 = nn.Conv1d(in_channels, 2*in_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(2*in_channels)

        self.resblock1 = BasicResidualBlock(2*in_channels, 8*in_channels, kernel_size=3, stride=2, padding=1, 
                                            Conv = CoordConv1d, BatchNorm = nn.BatchNorm1d, Activ = nn.ReLU)
        
        self.resblock2 = BasicResidualBlock(8*in_channels, 32*in_channels, kernel_size=3, stride=2, padding=1, 
                                            Conv = CoordConv1d, BatchNorm = nn.BatchNorm1d, Activ = nn.ReLU)
        
        self.resblock3 = BasicResidualBlock(32*in_channels, 64*in_channels, kernel_size=3, stride=2, padding=1, 
                                            Conv = CoordConv1d, BatchNorm = nn.BatchNorm1d, Activ = nn.ReLU)
        
        self.resblock4 = BasicResidualBlock(64*in_channels, 64*in_channels, kernel_size=3, stride=2, padding=1, 
                                            Conv = CoordConv1d, BatchNorm = nn.BatchNorm1d, Activ = nn.ReLU)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(64*in_channels*ceil(signal_length/16), 32*out_channels)
        self.fc2 = nn.Linear(32*out_channels, out_channels)

        self.dropout = nn.Dropout(p=0.3) # help avoid overfitting

    def forward(self, z):
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.resblock1(z)
        z = self.resblock2(z)
        z = self.resblock3(z)
        z = self.resblock4(z)

        z = torch.flatten(z, 1) 

        z = self.fc1(z)  
        z = self.dropout(z)
        z = self.sigmoid(z)

        z = self.fc2(z)  

        return z
    

class PoolGaussNet(nn.Module):
    '''
    Simple architecture no pooling. Positional information is concatenated to the input at the start.
    '''
    def __init__(self, in_channels, out_channels, signal_length):

        super(PoolGaussNet, self).__init__()

        # Initial 1D convolution layer
        self.conv1 = nn.Conv1d(in_channels, 4*in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(4*in_channels)
        self.conv2 = nn.Conv1d(4*in_channels, 8*in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(8*in_channels)
        self.conv3 = nn.Conv1d(8*in_channels, 16*in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(16*in_channels)
        self.conv4 = nn.Conv1d(16*in_channels, 16*in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(16*in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool1d(kernel_size = signal_length, return_indices=True)

        self.fc1 = nn.Linear(2*64*in_channels, 32*in_channels)
        self.fc2 = nn.Linear(32*in_channels, 8*out_channels)
        self.fc3 = nn.Linear(8*out_channels, out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, z):
        signal_length = z.size(-1)
        # idx = 
        # z = torch.concat([z, x], dim=1)

        z = self.conv1(z)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.conv2(z)
        z = self.bn2(z)
        z = self.relu(z)

        z = self.conv3(z)
        z = self.bn3(z)
        z = self.relu(z)
        
        z = self.conv4(z)
        z = self.bn4(z)
        z = self.relu(z)

        z, idx = self.pool(z) # idx returned by MaxPool2d will contain the flattened indices of the matrix (C, D).
        idx = idx % signal_length # We require indices only along D dimension.
        x = idx / signal_length # normalise to [0, 1]

        z = torch.stack([z, x], dim=-1)
        z = torch.flatten(z, 1)  # (N, 2*32*C)

        z = self.fc1(z)  
        z = self.dropout(z)
        z = self.relu(z)

        z = self.fc2(z)  
        z = self.dropout(z)
        z = self.relu(z)

        z = self.fc3(z)  

        return z # Output shape: (batch_size, (2 * in_channels + 1) * num_gaussians)
    
class ResGaussNet(nn.Module):
    '''
    The architecture is based on ResNet.

    Input data:
      Spectra containing tied spectral lines.
      shape (N, C, D) = (batch size, spectral line count, specral line data length)

    Output data:
      Inferred parameters for each Gaussian.
      shape (N, (2*C + 1)*GpL) = (batch size, (2 * lines + 1) * gaussians)
    '''
    def __init__(self, in_channels, out_channels, signal_length, num_blocks=1, hidden_channels=32):

        super(ResGaussNet, self).__init__()

        # Initial 1D convolution layer
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.res_layers = nn.Sequential(
              *(BasicResidualBlock(hidden_channels * 2**i, hidden_channels * 2**(i+1), stride=2) for i in range(num_blocks))
            )

        # Global average pooling and final regression layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)  # or another suitable rate
        self.fc = nn.Linear(hidden_channels * 2**num_blocks, out_channels)

    def forward(self, z):
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.res_layers(z)

        z = self.avg_pool(z)
        z = torch.flatten(z, 1)  # Flatten to (batch_size, hidden_channels)
        z = self.dropout(z)
        z = self.fc(z)  # Output shape: (batch_size, out_channels)

        return z

class SymGaussNet(nn.Module):
    '''
    The architecture is symmetric on the channels, meaning that a permutation of
    input spectral lines will give a permutation of output channels. For this we
    apply identical operatons on each channel independently, an finally combine
    the outputs.

    Input data:
      Spectra containing tied spectral lines.
      shape (N, C, D) = (batch size, spectral line count, specral line data length)

    Output data:
      Inferred parameters for each Gaussian.
      shape (N, (2*C + 1)*GpL) = (batch size, (2 * lines + 1) * gaussians)
    '''
    def __init__(self, in_channels, out_channels, signal_length):
        super(SymGaussNet, self).__init__()

        # Apply convolution across the D dimension with shared weights for each channel in C:
        # Conv2d with kernel shape (1, kernel_size) will convolve over D, independently for each channel
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 7), padding=(0, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 2), bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size = (1, signal_length), return_indices=True)

        self.fc1 = nn.Linear(2*32*in_channels, 8*out_channels)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(8*out_channels, out_channels)

        # # Separate output heads for Gaussian parameters
        # self.fc_amplitude = nn.Linear(hidden_channels, num_gaussians)
        # self.fc_mean = nn.Linear(hidden_channels, 1)
        # self.fc_variance = nn.Linear(hidden_channels, num_gaussians)

    def forward(self, z):

        signal_length = z.size(-1)

        # Reshape input to (N, 1, C, D)
        z = z.unsqueeze(1)  # Add an extra dimension for 2D convolution

        z = self.conv1(z)
        z = self.relu(z)
        z = self.bn1(z)

        z = self.conv2(z)
        z = self.relu(z)
        z = self.bn2(z)

        z = self.conv3(z)
        z = self.relu(z)
        z = self.bn3(z)
        # Conv output is (N, 32, C, D)

        z, idx = self.pool(z) # idx returned by MaxPool2d will contain the flattened indices of the matrix (C, D).
        idx = idx % signal_length # We require indices only along D dimension.
        x = idx / idx.size(-1) # normalise
        x = x

        z = torch.stack([z, x], dim=-1)
        z = torch.flatten(z, 1)  # (N, 2*32*C)
        z = self.fc1(z)
        z = self.dropout(z)
        z = self.fc2(z)

        return z

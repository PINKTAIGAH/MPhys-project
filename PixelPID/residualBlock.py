import torch
import torch.nn as nn
from convBlock import *

class ResidualBlock3D(nn.Module):
    # class for non-downsizing residual block

    def __init__(self,channels, act="leaky", batchnorm=True):
        super().__init__()

        # Define convolutional blocks in residual blocks
        self.resBlock = nn.Sequential(
            ConvBlock3D(
                channels, 
                channels,
                act=act,
                batchnorm=batchnorm,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            ConvBlock3D(
                channels,
                channels,
                act="identity",
                batchnorm=batchnorm,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
    
    def forward(self, x):
        # Define operations to be made to input when object is called
        
        return x + self.resBlock(x) 


class DownsampleResidualBlock3D(nn.Module):
    # Basic Residual block

    def __init__(self, inChannels, outChannels, act="leaky", batchnorm=True):
        super().__init__()

        # Define convolutional blocks in residual blocks
        self.resBlockDown = nn.Sequential(
            ConvBlock3D(
                inChannels, 
                outChannels,
                act=act,
                batchnorm=batchnorm,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            ConvBlock3D(
                outChannels, 
                outChannels,
                act="identity",
                batchnorm=batchnorm,
                kernel_size=3,
                stride=1,
                padding=1
            ),
        )

        # Define normal downsample convolution for resudial calculation in input is downsamples
        self.resBlockDownSkip = nn.Sequential(
            ConvBlock3D(
                inChannels,
                outChannels,
                act= "identity",
                batchnorm=True,
                kernel_size=1,
                stride=2,
                padding=0
            ),      
        )
        
    def forward(self, x):
        # Define operations to be made to input when object is called
        
        return self.resBlockDownSkip(x) + self.resBlockDown(x)

class ResidualBlock2D(nn.Module):
    # class for non-downsizing residual block

    def __init__(self,channels, act="leaky", batchnorm=True):
        super().__init__()

        # Define convolutional blocks in residual blocks
        self.resBlock = nn.Sequential(
            ConvBlock2D(
                channels, 
                channels,
                act=act,
                batchnorm=batchnorm,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            ConvBlock2D(
                channels,
                channels,
                act="identity",
                batchnorm=batchnorm,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
    
    def forward(self, x):
        # Define operations to be made to input when object is called
        
        return x + self.resBlock(x) 


class DownsampleResidualBlock2D(nn.Module):
    # Basic Residual block

    def __init__(self, inChannels, outChannels, act="leaky", batchnorm=True):
        super().__init__()

        # Define convolutional blocks in residual blocks
        self.resBlockDown = nn.Sequential(
            ConvBlock2D(
                inChannels, 
                outChannels,
                act=act,
                batchnorm=batchnorm,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            ConvBlock2D(
                outChannels, 
                outChannels,
                act="identity",
                batchnorm=batchnorm,
                kernel_size=3,
                stride=1,
                padding=1
            ),
        )

        # Define normal downsample convolution for resudial calculation in input is downsamples
        self.resBlockDownSkip = nn.Sequential(
            ConvBlock2D(
                inChannels,
                outChannels,
                act= "identity",
                batchnorm=True,
                kernel_size=1,
                stride=2,
                padding=0
            ),      
        )
        
    def forward(self, x):
        # Define operations to be made to input when object is called

        return self.resBlockDownSkip(x) + self.resBlockDown(x)
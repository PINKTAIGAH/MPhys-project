import torch
import torch.nn as nn
from torch.nn.modules import padding

class _ConvBlock(nn.Module):

    def __init__(self, inChannels, outChannels, down=True, act="identity", **kwargs):
        super().__init__()

        # Define what activation function will be used
        match act:
            case "identity":
                activationFunction = nn.Identity()
            case "relu":
                activationFuncton = nn.ReLU()
            case "leaky":
                activationFunction = nn.LeakyReLU(negative_slope=0.2)

        # Define generic convolutional block and transpose convolutional block
        self.conv = nn.Sequential(
            nn.Conv2d(
                inChannels,
                outChannels,
                padding_mode="reflect",
                **kwargs,
            )
            if down else nn.ConvTranspose2d(
                inChannels,
                outChannels,
                **kwargs,
            ),
            nn.BatchNorm2d(outChannels),
            activationFunction,    
        )

    def forward(self, x):
        # Output result of conv block when class is called 
        
        return self.conv(x) 

class _ResudualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        # Define convolutional block used in the residual blocks
        self.resBlock = nn.Sequential(
            _ConvBlock(
                channels,
                channels,
                act="leaky",
                kernal_size=3,
                padding=1,
                stride=1
            ),
            _ConvBlock(
                channels,
                channels,
                act="identity",
                kernal_size=3,
                padding=1,
                stride=1
            ),
        )

    def forward(self, x):

        # Define operations to be made to input when class is called
        return x + self.resBlock(x)


class _DimentionalBlock(nn.Module):

    def __init__(self, inChannels, outChannels, **kwargs):

        super().__init()

        self.dimentionalBlock = nn.Sequential(
            _ConvBlock(
                inChannels,
                outChannels,
                act="leaky",
                kernal_size=3,
                stride=1,
                padding=1
            ),
            _ResudualBlock(
                outChannels
            ),
            _ResudualBlock(
                outChannels
            )
        )

    def forward(self, x):

        # Define operations to be made to input when class is called
        return self.dimentionalBlock(x)

    

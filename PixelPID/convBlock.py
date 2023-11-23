import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    # Class constaining basic convolutional block

    def __init__(self, inChannels, outChannels, down=True, act="relu", batchnorm=True, **kwargs):
        super().__init__()

        # Define activation function to be used in this block
        match act:
            case "identity":
                activationFunction = nn.Identity()
            case "relu":
                activationFunction = nn.ReLU()
            case "leaky":
                activationFunction = nn.LeakyReLU()
            case "gelu":
                activationFunction = nn.GELU()
            case _:
                raise Exception(f"{act} is not a recognised activation function for this class")

        # Define generic convolutional block / transpose convolutional block

        self.conv = nn.Sequential(
            nn.Conv3d(
                inChannels, 
                outChannels,
                padding_mode="reflect",
                **kwargs,
            ) 
            if down else nn.ConvTranspose3d(
                inChannels,
                outChannels,
                **kwargs,
            ),
            nn.BatchNorm3d(outChannels) if batchnorm else nn.Identity(),
            activationFunction,
        )

    def forward(self, x):
        # Output result of conv block when object is called

        return self.conv(x)

class ConvBlock2D(nn.Module):
    # Class constaining basic convolutional block

    def __init__(self, inChannels, outChannels, down=True, act="relu", batchnorm=True, **kwargs):
        super().__init__()

        # Define activation function to be used in this block
        match act:
            case "identity":
                activationFunction = nn.Identity()
            case "relu":
                activationFunction = nn.ReLU()
            case "leaky":
                activationFunction = nn.LeakyReLU()
            case "gelu":
                activationFunction = nn.GELU()
            case _:
                raise Exception(f"{act} is not a recognised activation function for this class")

        # Define generic convolutional block / transpose convolutional block

        self.conv = nn.Sequential(
            nn.Conv2d(
                inChannels, 
                outChannels,
                padding_mode="reflect",
                **kwargs,
            ) 
            if down else nn.ConvTranspose3d(
                inChannels,
                outChannels,
                **kwargs,
            ),
            nn.BatchNorm2d(outChannels) if batchnorm else nn.Identity(),
            activationFunction,
        )

    def forward(self, x):
        # Output result of conv block when object is called

        return self.conv(x)
import torch
import torch.nn as nn
from convBlock import ConvBlock
from linearBlock import LinearBlock
from residualBlock import *


class ResnetMedMnist(nn.Module):
    def __init__(self, imageChannels, numClasses, imageDimentions=(64, 64, 64),
                numFeatures=64, listResiduals=[3, 8,]):

        super().__init__()

        # Define initial block of the analiser
        self.initialLayer = nn.Sequential(
            ConvBlock(
                imageChannels,
                numFeatures,
                act="leaky",
                batchnorm=True,
                kernel_size=7,
                stride=2,
                padding=3,               
                bias=False,
            ),
            nn.BatchNorm3d(numFeatures),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1,),
        )

        # Define first residual block with 64 features
        # Note: we use actual number in list as there is no downsample res block
        resBlock64 = nn.Sequential(
            *[ResidualBlock(numFeatures, act="leaky", batchnorm=True) for _ in range(listResiduals[0])],
        )

        # Define second residual block with 128 features
        resBlock128 = nn.Sequential(
            DownsampleResidualBlock(numFeatures, numFeatures*2, act="leaky"),
            *[ResidualBlock(numFeatures*2, act="leaky", batchnorm=True) for _ in range(listResiduals[1]-1)],
        )

        self.resBlocksAll = nn.ModuleList([
            resBlock64,
            resBlock128,
        ])

        # We know that the hight and width of the latent tensor after all resnet is (B, 128, H/7, W/7, L/7)
        # Define number of nodes in linear layers
        productLatentDimentions = int((imageDimentions[0]/7) * (imageDimentions[1]/7) * (imageDimentions[2]/7))
        flattenedInFeatures = 128*productLatentDimentions
        
        self.denseBlocks = nn.ModuleList([
            LinearBlock(
                flattenedInFeatures,
                numClasses,
                act="identity",
                flatten=True,
                bias=True
            ),
        ])

    def forward(self, x):

        # Apply initial layer
        x = self.initialLayer(x)        # Size: (B, 64, H/2, W/2, L/2)

        # Apply all resnet layers
        for layer in self.resBlocksAll:
            x = layer(x)
        # Size: (B, 128, H/7, W/7, L/7)
        
        # Apply linear layers 
        for layer in self.denseBlocks:
            x = layer(x)

        return x
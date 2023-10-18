import torch
import torch.nn as nn
from convBlock import ConvBlock
from linearBlock import LinearBlock
from residualBlock import *

class Classifier3D(nn.Module):
    def __init__(self, imageChannels, numClasses, imageDimentions=(64, 64, 64),
                numFeatures=64, listResiduals=[3, 4, 6, 3]):

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

        # Define third residual block with 256 features
        resBlock256 = nn.Sequential(
            DownsampleResidualBlock(numFeatures*2, numFeatures*4, act="leaky"),
            *[ResidualBlock(numFeatures*4, act="leaky", batchnorm=True) for _ in range(listResiduals[2]-1)],
        )

        # Define third residual block with 512 features
        resBlock512 = nn.Sequential(
            DownsampleResidualBlock(numFeatures*4, numFeatures*8, act="leaky"),
            ResidualBlock(numFeatures*8, act="leaky", batchnorm=True),
            ResidualBlock(numFeatures*8, act="leaky", batchnorm=False)
        )

        self.resBlocksAll = nn.ModuleList([
            resBlock64,
            resBlock128,
            resBlock256,
            resBlock512
        ])

        # We know that the hight and width of the latent tensor after all resnet is (B, 512, H/32, W/32, L/32)
        # Define number of nodes in linear layers
        productLatentDimentions = int((imageDimentions[0]/32) * (imageDimentions[1]/32) * (imageDimentions[2]/32))
        flattenedInFeatures = 512*productLatentDimentions
        
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
        # Size: (B, 512, H/32, W/32, L/32)
        
        # Apply linear layers 
        for layer in self.denseBlocks:
            x = layer(x)

        return x
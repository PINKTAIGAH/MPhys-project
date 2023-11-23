import torch
import torch.nn as nn
from convBlock import ConvBlock
from linearBlock import LinearBlock
from residualBlock import *


"""
Model based on C. Adams et al. Journal of Instrumentation, Volume 15, April 2020
"""

class sparceNetPID(nn.Module):
    def __init__(self, imageChannels, numClasses, imageDimentions=(64, 64, 64),
                numFeatures=32, numResiduals=2):

        super().__init__()

        # Define initial block of the analiser
        self.initialLayer = nn.Sequential(
            ConvBlock3D(
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
        resBlock1 = ResidualBlock3D(numFeatures, act="leaky", batchnorm=True)
        
        # Define second residual block with 128 features
        resBlock2 = nn.Sequential(
            DownsampleResidualBlock3D(numFeatures, numFeatures*2, act="leaky"),
            *[ResidualBlock3D(numFeatures*2, act="leaky", batchnorm=True) for _ in range(numResiduals)],
        )

        # Define third residual block with 256 features
        resBlock3 = nn.Sequential(
            DownsampleResidualBlock3D(numFeatures*2, numFeatures*4, act="leaky"),
            *[ResidualBlock3D(numFeatures*4, act="leaky", batchnorm=True) for _ in range(numResiduals)],
        )

        # Define fourth residual block with 512 features
        resBlock4 = nn.Sequential(
            DownsampleResidualBlock3D(numFeatures*4, numFeatures*8, act="leaky"),
            *[ResidualBlock3D(numFeatures*8, act="leaky", batchnorm=True) for _ in range(numResiduals)],
        )

        # Define fifth residual block with 512 features
        resBlock5 = nn.Sequential(
            DownsampleResidualBlock3D(numFeatures*8, numFeatures*16, act="leaky"),
            *[ResidualBlock3D(numFeatures*16, act="leaky", batchnorm=True) for _ in range(numResiduals)],
        )

        # Define sixth residual block with 512 features
        resBlock6 = nn.Sequential(
            DownsampleResidualBlock3D(numFeatures*16, numFeatures*32, act="leaky"),
            *[ResidualBlock3D(numFeatures*32, act="leaky", batchnorm=True) for _ in range(numResiduals)],
        )

        # Define seventh residual block with 512 features
        resBlock7 = nn.Sequential(
            DownsampleResidualBlock3D(numFeatures*32, numFeatures*64, act="leaky"),
            *[ResidualBlock3D(numFeatures*64, act="leaky", batchnorm=True) for _ in range(numResiduals)],
        )

        self.resBlocksAll = nn.ModuleList([
            resBlock1,
            resBlock2,
            resBlock3,
            resBlock4,
            resBlock5,
            resBlock6,
            resBlock7,
        ])
        
        # We know that the hight and width of the latent tensor after all resnet is (B, 512, H/2**7, W/2**7, L/2**7)
        # Define number of nodes in linear layers
        productLatentDimentions = int((imageDimentions[0]/256) * (imageDimentions[1]/256) * (imageDimentions[2]/236))
        flattenedInFeatures = numFeatures*64*productLatentDimentions
        
        self.denseBlocks = nn.ModuleList([
            LinearBlock(
                flattenedInFeatures,
                numClasses,
                act="softmax",
                flatten=True,
                bias=True
            ),
        ])


    def forward(self, x):

        # Apply initial layer
        x = self.initialLayer(x)        # Size: (B, 32, H/2, W/2, L/2)

        # Apply all resnet layers
        for layer in self.resBlocksAll:
            x = layer(x)
 
        # Apply linear layers 
        for layer in self.denseBlocks:
            x = layer(x)

        return x


def test():
    DEVICE = "cpu"
    CHANNELS = 1
    N_CLASSES = 4
    IMAGE_DIMENTIONS = (512, 512, 512)
    input = torch.rand((1, 1, *IMAGE_DIMENTIONS)).to(DEVICE)
    model = sparceNetPID(CHANNELS, N_CLASSES, IMAGE_DIMENTIONS).to(DEVICE)
    output = model(input)
    print(output.shape)

if __name__ == "__main__":
    test()
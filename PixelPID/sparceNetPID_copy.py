import torch
import torch.nn as nn
from convBlock import *
from linearBlock import LinearBlock
from residualBlock import *

class SparceNetPID(nn.Module):
    def __init__(self,imageChannels, numClasses, imageDimentions=(1080, 2048), 
                 numFeatures=32, numResiduals=2,):
        super().__init__()

        # Define number of classes
        self.numClasses = numClasses

        # Define initial block
        initialBlock = nn.Sequential(
            ConvBlock2D(
                imageChannels, 
                numFeatures,
                act="leaky",
                batchnorm=True,
                kernel_size=5,
                stride=1, 
                padding=2,
                bias=False
            )
        )


        # Define first resblock
        resBlock1 = ResidualBlock2D(numFeatures, act="leaky", batchnorm=True)

        # Define second resblock
        resBlock2 = nn.Sequential(
            ConvBlock2D(numFeatures, 2*numFeatures, act="leaky", batchnorm=True, kernel_size=3, stride=2, padding=1, bias=False),
            *[ResidualBlock2D(2*numFeatures, act="leaky", batchnorm=True) for _ in range(numResiduals)],
        )

        # Define third resblock
        resBlock3 = nn.Sequential(
            ConvBlock2D(2*numFeatures, 3*numFeatures, act="leaky", batchnorm=True, kernel_size=3, stride=2, padding=1, bias=False),
            *[ResidualBlock2D(3*numFeatures, act="leaky", batchnorm=True) for _ in range(numResiduals)],
        )

        # Define fourth resblock
        resBlock4 = nn.Sequential(
            ConvBlock2D(3*numFeatures, 4*numFeatures, act="leaky", batchnorm=True, kernel_size=3, stride=2, padding=1, bias=False),
            *[ResidualBlock2D(4*numFeatures, act="leaky", batchnorm=True) for _ in range(numResiduals)],
        )

        # Define Downsample convolution
        downsample = nn.Sequential(
            ConvBlock2D(4*numFeatures, 5*numFeatures, act="leaky", kernel_size=3, stride=2, padding=1, bias=False)
        )

        # Construct first stage of convolutions to be applied ot each individual 2d image
        self.totemPoleBlock = nn.ModuleList([
            initialBlock,
            resBlock1,
            resBlock2,
            resBlock3,
            resBlock4,
            downsample,
        ])

        # Define fifth resblock
        resBlock5 = nn.Sequential(
            *[ResidualBlock2D(5*numFeatures, "leaky", batchnorm=True) for _ in range(numResiduals+1)],
        )

        # Define sixth resblock
        resBlock6 = nn.Sequential(
            ConvBlock2D(5*numFeatures, 6*numFeatures, act="leaky", batchnorm=True, kernel_size=3, stride=2, padding=1, bias=False),
            *[ResidualBlock2D(6*numFeatures, "leaky", batchnorm=True) for _ in range(numResiduals)],
        )

        # Define seventh resblock
        resBlock7 = nn.Sequential(
            ConvBlock2D(6*numFeatures, 7*numFeatures, act="leaky", batchnorm=True, kernel_size=3, stride=2, padding=1, bias=False),
            *[ResidualBlock2D(7*numFeatures, "leaky", batchnorm=True) for _ in range(numResiduals)],
        )

        # Define bottleneck
        bottleneck = ConvBlock2D(7*numFeatures, imageChannels, act="leaky", kernel_size=3, stride=1, padding=1, bias=False)

        # Define second stage of convolutions
        self.concatonatedBlock = nn.ModuleList([
            resBlock5,
            resBlock6,
            resBlock7,
            bottleneck,
        ])

    def getDenseLayer(self, inFeatures, outFeatures):
        """
        Construct a dense layer with correct dimentions
        """

        denseLayer = nn.Sequential(
            LinearBlock(
                inFeatures,
                outFeatures,
                act = "identity",
                flatten = True,
                bias = True, 
            )
        )

        return denseLayer 

    def forward(self, x, device="cpu"):
        # Apply siamese tower block to each image plane
        for layer in self.totemPoleBlock:
            x = layer(x)
        
        # Apply second convolutinal blocks
        for layer in self.concatonatedBlock:
            x = layer(x)
        # Get dense layer and apply it
        outputFlattenedShape = torch.prod(torch.tensor(x.shape[1:])).item()
        denseLayer = self.getDenseLayer(outputFlattenedShape, self.numClasses).to(device)
        x = denseLayer(x)

        return x

def test():
    device = "cuda"
    channels = 1
    n_classes = 4
    inputs = torch.randn(1, 1, 640, 1024).to(device) 
    model = SparceNetPID(channels, n_classes, imageDimentions=(540, 1024)).to(device)

    model(inputs, device)

if __name__ == "__main__":
    test()


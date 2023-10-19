import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    # Class containing basic Linear Block

    def __init__(self, inFeatures, outFeatures, act="relu", flatten=False, **kwargs):
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
            case "sigmoid":
                activationFunction = nn.Sigmoid()
            case "softmax":
                activationFunction = nn.Softmax(dim=0)
            case _:
                raise Exception(f"{act} is not a recognised activation function for this class")

        self.linear = nn.Sequential(
            nn.Flatten(start_dim=1) if flatten else nn.Identity(), 
            nn.Linear(
                inFeatures,
                outFeatures,
                **kwargs,
            ),
            activationFunction,
        )

    def forward(self, x):
        # Output result of conv block when object is called

        return self.linear(x)
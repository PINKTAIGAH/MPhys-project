import torch
import torch.nn as nn

class _ConvBlock(nn.Module):

    def __init__(self, inChannels, outChannels, down=True, act="relu", **kwargs):
        super().__init__()

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
            nn.ReLU() if act=="relu" else  nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        # Output result of conv block when class is called 
        
        return self.conv(x) 

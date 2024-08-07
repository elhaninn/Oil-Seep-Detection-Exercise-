import torch
import torch.nn as nn
from typing import List

class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList([
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            self.conv_block(512, 1024)
        ])
        
        self.decoder = nn.ModuleList([
            self.up_conv(1024, 512),
            self.conv_block(1024, 512),
            self.up_conv(512, 256),
            self.conv_block(512, 256),
            self.up_conv(256, 128),
            self.conv_block(256, 128),
            self.up_conv(128, 64),
            self.conv_block(128, 64)
        ])
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels: int, out_channels: int) -> nn.ConvTranspose2d:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_outputs: List[torch.Tensor] = []
        
        for encode in self.encoder:
            x = encode(x)
            enc_outputs.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        
        for i, decode in enumerate(self.decoder[::2]):
            x = decode(x)
            x = torch.cat([x, enc_outputs[-(i+1)]], dim=1)
            x = self.decoder[2*i + 1](x)
        
        return self.final_conv(x)

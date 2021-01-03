import torch.nn as nn
from easydict import EasyDict
from models.layers.unet_layers import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, options: EasyDict):
        super(UNet, self).__init__()
        self.n_channels = options.model.in_channel
        self.n_classes = options.model.out_channel
        self.bilinear = options.model.bilinear_up

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, batch):
        x1 = self.inc(batch["input_image"])
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return {"pred_error_image": logits}

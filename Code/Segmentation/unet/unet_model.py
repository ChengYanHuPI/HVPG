""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from torch.nn import functional as F
from .unet_parts import DoubleConv, Down, Up, Up_part, OutConv, SE_Block, Residual


class UNet_4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.seb1 = SE_Block(16)
        self.residual1 = Residual(16, 16)
        self.down1 = Down(16, 32)

        self.seb2 = SE_Block(32)
        self.residual2 = Residual(32, 32)
        self.down2 = Down(32, 64)

        self.seb3 = SE_Block(64)
        self.residual3 = Residual(64, 64)
        self.down3 = Down(64, 128)

        self.seb4 = SE_Block(128)
        self.residual4 = Residual(128, 128)
        self.down4 = Down(128, 256)

        self.up1 = Up(256, 128, bilinear)
        self.seb5 = SE_Block(128)
        self.residual5 = Residual(128, 128)

        self.up1_2 = Up_part(256, 16, 8)
        self.seb5_2 = SE_Block(16)
        self.residual5_2 = Residual(16, 16)

        self.up2 = Up(128, 64, bilinear)
        self.seb6 = SE_Block(64)
        self.residual6 = Residual(64, 64)

        self.up2_2 = Up_part(128, 16, 4)
        self.seb6_2 = SE_Block(16)
        self.residual6_2 = Residual(16, 16)

        self.up3 = Up(64, 32, bilinear)
        self.seb7 = SE_Block(32)
        self.residual7 = Residual(32, 32)

        self.up3_2 = Up_part(64, 16, 2)
        self.seb7_2 = SE_Block(16)
        self.residual7_2 = Residual(16, 16)

        self.up4 = Up(32, 16, bilinear)
        self.seb8 = SE_Block(16)
        self.residual8 = Residual(16, 16)

        self.outc1 = OutConv(16, n_classes)
        self.outc2 = OutConv(16, n_classes)
        self.outc3 = OutConv(16, n_classes)
        self.outc4 = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.seb1(x1)
        x1 = self.residual1(x1)
        x2 = self.down1(x1)

        x2 = self.seb2(x2)
        x2 = self.residual2(x2)
        x3 = self.down2(x2)

        x3 = self.seb3(x3)
        x3 = self.residual3(x3)
        x4 = self.down3(x3)

        x4 = self.seb4(x4)
        x4 = self.residual4(x4)
        x5 = self.down4(x4)

        x_up1 = self.up1(x5, x4)
        x_up1 = self.seb5(x_up1)
        x_up1 = self.residual5(x_up1)

        x_1 = self.up1_2(x5, x1)
        x_1 = self.seb5_2(x_1)
        x_1 = self.residual5_2(x_1)
        logit1 = self.outc1(x_1)

        x_up2 = self.up2(x_up1, x3)
        x_up2 = self.seb6(x_up2)
        x_up2 = self.residual6(x_up2)

        x_2 = self.up2_2(x_up1, x1)
        x_2 = self.seb6_2(x_2)
        x_2 = self.residual6_2(x_2)
        logit2 = self.outc2(x_2)

        x_up3 = self.up3(x_up2, x2)
        x_up3 = self.seb7(x_up3)
        x_up3 = self.residual7(x_up3)

        x_3 = self.up3_2(x_up2, x1)
        x_3 = self.seb7_2(x_3)
        x_3 = self.residual7_2(x_3)
        logit3 = self.outc3(x_3)

        x_up4 = self.up4(x_up3, x1)
        x_up4 = self.seb8(x_up4)
        x_up4 = self.residual8(x_up4)

        logit4 = self.outc4(x_up4)

        return logit1, logit2, logit3, logit4


class UNet_3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.seb1 = SE_Block(16)
        self.residual1 = Residual(16, 16)
        self.down1 = Down(16, 32)

        self.seb2 = SE_Block(32)
        self.residual2 = Residual(32, 32)
        self.down2 = Down(32, 64)

        self.seb3 = SE_Block(64)
        self.residual3 = Residual(64, 64)
        self.down3 = Down(64, 128)

        self.up1 = Up(128, 64, bilinear)
        self.seb5 = SE_Block(64)
        self.residual5 = Residual(64, 64)

        self.up1_2 = Up_part(128, 16, 4)
        self.seb5_2 = SE_Block(16)
        self.residual5_2 = Residual(16, 16)

        self.up2 = Up(64, 32, bilinear)
        self.seb6 = SE_Block(32)
        self.residual6 = Residual(32, 32)

        self.up2_2 = Up_part(64, 16, 2)
        self.seb6_2 = SE_Block(16)
        self.residual6_2 = Residual(16, 16)

        self.up3 = Up(32, 16, bilinear)
        self.seb7 = SE_Block(16)
        self.residual7 = Residual(16, 16)

        self.outc1 = OutConv(16, n_classes)
        self.outc2 = OutConv(16, n_classes)
        self.outc3 = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.seb1(x1)
        x1 = self.residual1(x1)
        x2 = self.down1(x1)

        x2 = self.seb2(x2)
        x2 = self.residual2(x2)
        x3 = self.down2(x2)

        x3 = self.seb3(x3)
        x3 = self.residual3(x3)
        x4 = self.down3(x3)

        x_up1 = self.up1(x4, x3)
        x_up1 = self.seb5(x_up1)
        x_up1 = self.residual5(x_up1)

        x_1 = self.up1_2(x4, x1)
        x_1 = self.seb5_2(x_1)
        x_1 = self.residual5_2(x_1)
        logit1 = self.outc1(x_1)

        x_up2 = self.up2(x_up1, x2)
        x_up2 = self.seb6(x_up2)
        x_up2 = self.residual6(x_up2)

        x_2 = self.up2_2(x_up1, x1)
        x_2 = self.seb6_2(x_2)
        x_2 = self.residual6_2(x_2)
        logit2 = self.outc2(x_2)

        x_up3 = self.up3(x_up2, x1)
        x_up3 = self.seb7(x_up3)
        x_up3 = self.residual7(x_up3)

        logit3 = self.outc3(x_up3)

        return logit1, logit2, logit3


class UNet_2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.seb1 = SE_Block(16)
        self.residual1 = Residual(16, 16)
        self.down1 = Down(16, 32)

        self.seb2 = SE_Block(32)
        self.residual2 = Residual(32, 32)
        self.down2 = Down(32, 64)

        self.up1 = Up(64, 32, bilinear)
        self.seb5 = SE_Block(32)
        self.residual5 = Residual(32, 32)

        self.up1_2 = Up_part(64, 16, 2)
        self.seb5_2 = SE_Block(16)
        self.residual5_2 = Residual(16, 16)

        self.up2 = Up(32, 16, bilinear)
        self.seb6 = SE_Block(16)
        self.residual6 = Residual(16, 16)

        self.outc1 = OutConv(16, n_classes)
        self.outc2 = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.seb1(x1)
        x1 = self.residual1(x1)
        x2 = self.down1(x1)

        x2 = self.seb2(x2)
        x2 = self.residual2(x2)
        x3 = self.down2(x2)

        x_up1 = self.up1(x3, x2)
        x_up1 = self.seb5(x_up1)
        x_up1 = self.residual5(x_up1)

        x_1 = self.up1_2(x3, x1)
        x_1 = self.seb5_2(x_1)
        x_1 = self.residual5_2(x_1)
        logit1 = self.outc1(x_1)

        x_up2 = self.up2(x_up1, x1)
        x_up2 = self.seb6(x_up2)
        x_up2 = self.residual6(x_up2)

        logit2 = self.outc2(x_up2)

        return logit1, logit2


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.seb1 = SE_Block(32)
        self.residual1 = Residual(32, 32)
        self.down1 = Down(32, 64)

        self.seb2 = SE_Block(64)
        self.residual2 = Residual(64, 64)
        self.down2 = Down(64, 128)

        self.seb3 = SE_Block(128)
        self.residual3 = Residual(128, 128)
        self.down3 = Down(128, 256)

        self.seb4 = SE_Block(256)
        self.residual4 = Residual(256, 256)
        self.down4 = Down(256, 512)

        self.up1 = Up(512, 256, bilinear)
        self.seb5 = SE_Block(256)
        self.residual5 = Residual(256, 256)

        self.up2 = Up(256, 128, bilinear)
        self.seb6 = SE_Block(128)
        self.residual6 = Residual(128, 128)

        self.up3 = Up(128, 64, bilinear)
        self.seb7 = SE_Block(64)
        self.residual7 = Residual(64, 64)

        self.up4 = Up(64, 32, bilinear)
        self.seb8 = SE_Block(32)
        self.residual8 = Residual(32, 32)

        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.seb1(x1)
        x1 = self.residual1(x1)
        x2 = self.down1(x1)

        x2 = self.seb2(x2)
        x2 = self.residual2(x2)
        x3 = self.down2(x2)

        x3 = self.seb3(x3)
        x3 = self.residual3(x3)
        x4 = self.down3(x3)

        x4 = self.seb4(x4)
        x4 = self.residual4(x4)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.seb5(x)
        x = self.residual5(x)

        x = self.up2(x, x3)
        x = self.seb6(x)
        x = self.residual6(x)

        x = self.up3(x, x2)
        x = self.seb7(x)
        x = self.residual7(x)

        x = self.up4(x, x1)
        x = self.seb8(x)
        x = self.residual8(x)

        logits = self.outc(x)
        return logits


class WNet_2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(WNet_2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.seb1 = SE_Block(16)
        self.residual1 = Residual(16, 16)
        self.down1 = Down(16, 32)

        self.seb2 = SE_Block(32)
        self.residual2 = Residual(32, 32)
        self.down2 = Down(32, 64)

        self.up1 = Up(64, 32, bilinear)
        self.seb3 = SE_Block(32)
        self.residual3 = Residual(32, 32)

        self.up1_2 = Up_part(64, 16, 2)
        self.seb4_2 = SE_Block(16)
        self.residual4_2 = Residual(16, 16)

        self.up2 = Up(32, 16, bilinear)
        self.seb5 = SE_Block(16)
        self.residual5 = Residual(16, 16)

        self.outc1 = OutConv(16, n_classes)

        self.down3 = Down(16, 32)
        self.seb6 = SE_Block(32)
        self.residual6 = Residual(32, 32)
        self.down4 = Down(32, 64)

        self.up3 = Up(64, 32, bilinear)
        self.seb7 = SE_Block(32)
        self.residual7 = Residual(32, 32)

        self.up2_2 = Up_part(64, 16, 2)
        self.seb8_2 = SE_Block(16)
        self.residual8_2 = Residual(16, 16)

        self.up4 = Up(32, 16, bilinear)
        self.seb9 = SE_Block(16)
        self.residual9 = Residual(16, 16)

        self.outc2 = OutConv(16, n_classes)

        # self.inc = DoubleConv(n_channels, 64)
        # self.seb1 = SE_Block(64)
        # self.residual1 = Residual(64, 64)
        # self.down1 = Down(64, 128)

        # self.seb2 = SE_Block(128)
        # self.residual2 = Residual(128, 128)
        # self.down2 = Down(128, 256)

        # self.up1 = Up(256, 128, bilinear)
        # self.seb3 = SE_Block(128)
        # self.residual3 = Residual(128, 128)

        # self.up1_2 = Up_part(256, 64, 2)
        # self.seb4_2 = SE_Block(64)
        # self.residual4_2 = Residual(64, 64)

        # self.up2 = Up(128, 64, bilinear)
        # self.seb5 = SE_Block(64)
        # self.residual5 = Residual(64, 64)

        # self.outc1 = OutConv(64, n_classes)

        # self.down3 = Down(64, 128)
        # self.seb6 = SE_Block(128)
        # self.residual6 = Residual(128, 128)
        # self.down4 = Down(128, 256)

        # self.up3 = Up(256, 128, bilinear)
        # self.seb7 = SE_Block(128)
        # self.residual7 = Residual(128, 128)

        # self.up2_2 = Up_part(256, 64, 2)
        # self.seb8_2 = SE_Block(64)
        # self.residual8_2 = Residual(64, 64)

        # self.up4 = Up(128, 64, bilinear)
        # self.seb9 = SE_Block(64)
        # self.residual9 = Residual(64, 64)

        # self.outc2 = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.seb1(x1)
        x1 = self.residual1(x1)
        x2 = self.down1(x1)

        x2 = self.seb2(x2)
        x2 = self.residual2(x2)
        x3 = self.down2(x2)

        x_up1 = self.up1(x3, x2)
        x_up1 = self.seb3(x_up1)
        x_up1 = self.residual3(x_up1)

        x_1 = self.up1_2(x3, x1)
        x_1 = self.seb4_2(x_1)
        x_1 = self.residual4_2(x_1)

        x_up2 = self.up2(x_up1, x1)
        x_up2 = self.seb5(x_up2)
        x_up2 = self.residual5(x_up2)

        logit1 = self.outc1(x_up2)

        x4 = self.down3(x_up2)
        x4 = self.seb6(x4)
        x4 = self.residual6(x4)
        x5 = self.down4(x4)

        x_up3 = self.up3(x5, x4)
        x_up3 = self.seb7(x_up3)
        x_up3 = self.residual7(x_up3)

        x_2 = self.up2_2(x5, x_up2)
        x_2 = self.seb8_2(x_2)
        x_2 = self.residual8_2(x_2)

        x_up4 = self.up4(x_up3, x1)
        x_up4 = self.seb9(x_up4)
        x_up4 = self.residual9(x_up4)

        logit2 = self.outc2(x_up2)

        return logit1, logit2


# # Basic convolution blocks
# class Conv(nn.Module):
#     def __init__(self, C_in, C_out):
#         super(Conv, self).__init__()
#         self.layer = nn.Sequential(

#             nn.Conv2d(C_in, C_out, 3, 1, 1),
#             nn.BatchNorm2d(C_out),
#             # Prevent overfitting
#             nn.Dropout(0.3),
#             nn.LeakyReLU(),

#             nn.Conv2d(C_out, C_out, 3, 1, 1),
#             nn.BatchNorm2d(C_out),
#             # Prevent overfitting
#             nn.Dropout(0.4),
#             nn.LeakyReLU(),
#         )

#     def forward(self, x):
#         return self.layer(x)

# # Downsampling module
# class DownSampling(nn.Module):
#     def __init__(self, C):
#         super(DownSampling, self).__init__()
#         self.Down = nn.Sequential(
#             # 2X downsampling using convolution with the same number of channels
#             nn.Conv2d(C, C, 3, 2, 1),
#             nn.LeakyReLU()
#         )

#     def forward(self, x):
#         return self.Down(x)

# # Up-sampling module
# class UpSampling(nn.Module):

#     def __init__(self, C):
#         super(UpSampling, self).__init__()
#         # Feature map size is expanded by 2 times and the number of channels is halved
#         self.Up = nn.Conv2d(C, C // 2, 1, 1)

#     def forward(self, x, r):
#         # Downsampling using neighborhood interpolation
#         up = F.interpolate(x, scale_factor=2, mode="nearest")
#         x = self.Up(up)
#         # splicing, the current upsampling, and the previous downsampling process
#         return torch.cat((x, r), 1)

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()

#         # 4 times downsampling
#         self.C1 = Conv(3, 16)
#         self.D1 = DownSampling(16)
#         self.C2 = Conv(16, 32)
#         self.D2 = DownSampling(32)
#         self.C3 = Conv(32, 64)
#         self.D3 = DownSampling(64)
#         self.C4 = Conv(64, 128)
#         self.D4 = DownSampling(128)
#         self.C5 = Conv(128, 256)

#         # 4 times up-sampling
#         self.U1 = UpSampling(256)
#         self.C6 = Conv(256, 128)
#         self.U2 = UpSampling(128)
#         self.C7 = Conv(128, 64)
#         self.U3 = UpSampling(64)
#         self.C8 = Conv(64, 32)
#         self.U4 = UpSampling(32)
#         self.C9 = Conv(32, 16)

#         # self.Th = torch.nn.Sigmoid()
#         self.pred = torch.nn.Conv2d(16, 3, 3, 1, 1)

#     def forward(self, x):
#         # Downsampling section
#         R1 = self.C1(x)
#         R2 = self.C2(self.D1(R1))
#         R3 = self.C3(self.D2(R2))
#         R4 = self.C4(self.D3(R3))
#         Y1 = self.C5(self.D4(R4))

#         # Up-sampling section
#         # Need to be spliced together when upsampling
#         O1 = self.C6(self.U1(Y1, R4))
#         O2 = self.C7(self.U2(O1, R3))
#         O3 = self.C8(self.U3(O2, R2))
#         O4 = self.C9(self.U4(O3, R1))

#         # Output prediction, where the size is the same as the input
#         # You can gouge out the middle of the downsampling and then splice it, so that the modified output will be smaller
#         return self.Th(self.pred(O4))
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturePyrmaid(nn.Module):
    def __init__(self, in_channel=16):
        super(FeaturePyrmaid, self).__init__()

        self.out1 = nn.Sequential(nn.Conv2d(in_channel*3, in_channel * 4, kernel_size=3,
                                            stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channel * 4),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(in_channel * 4, in_channel * 4, kernel_size=1,
                                            stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(in_channel * 4),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  )

        self.out2 = nn.Sequential(nn.Conv2d(in_channel * 4, in_channel * 7, kernel_size=3,
                                            stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channel * 7),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(in_channel * 7, in_channel * 7, kernel_size=1,
                                            stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(in_channel * 7),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  )

        self.out3 = nn.Sequential(nn.Conv2d(in_channel * 7, in_channel * 10, kernel_size=3,
                                            stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channel * 10),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(in_channel * 10, in_channel * 10, kernel_size=1,
                                            stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(in_channel * 10),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  )

    def forward(self, x):
        # x: [B, 32, H, W]
        out1 = self.out1(x)  # [B, 64, H/2, W/2]
        out2 = self.out2(out1)  # [B, 128, H/4, W/4]
        out3 = self.out3(out2)
        return [x, out1, out2, out3]


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                               stride=2, padding=1)

        if self.concat:

            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3,
                                       stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class GANetFeature(nn.Module):
    """Height and width need to be divided by 48, downsampled by 1/3"""

    def __init__(self):
        super(GANetFeature, self).__init__()



        self.conv_start = nn.Sequential(
            BasicConv(3, 48, kernel_size=3, padding=1),
            BasicConv(48, 48, kernel_size=5, stride=4, padding=2),
            BasicConv(48, 48, kernel_size=3, padding=1))

        self.conv1a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(64, 80, kernel_size=3, stride=2, padding=1)


        self.conv3a = BasicConv(80, 112, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(112, 144, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(144, 112, deconv=True)
        self.deconv3a = Conv2x(112, 80, deconv=True)
        self.deconv2a = Conv2x(80, 64, deconv=True)
        self.deconv1a = Conv2x(64, 48, deconv=True)

        self.conv1b = Conv2x(48, 64)
        self.conv2b = Conv2x(64, 80)


        self.conv3b = Conv2x(80, 112)
        self.conv4b = Conv2x(112, 144)

        self.deconv4b = Conv2x(144, 112, deconv=True)
        self.deconv3b = Conv2x(112, 80, deconv=True)
        self.deconv2b = Conv2x(80, 64, deconv=True)
        self.deconv1b = Conv2x(64, 48, deconv=True)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 48, H/4, W/4]

        return x
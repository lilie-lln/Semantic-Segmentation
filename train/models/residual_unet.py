import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.identity_conv = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.identity_conv is not None:
            identity = self.identity_conv(identity)

        out += identity 
        out = self.relu(out)
        return out

class upconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upconv, self).__init__()
        self.conv = ResBlock(in_channels, out_channels)
        self.upconv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, inputs_R, inputs_U):
        outputs_U = self.upconv1(inputs_U)
        offset = outputs_U.size()[-1] - inputs_R.size()[-1]
        pad = [offset // 2, offset - offset // 2, offset // 2, offset - offset // 2]
        outputs_R = F.pad(inputs_R, pad)
        return self.conv(torch.cat((outputs_U, outputs_R), dim=1))

class Unet(nn.Module):
    def __init__(self, in_channels, n_classes=1):
        super(Unet, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        self.conv1 = ResBlock(in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = ResBlock(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = ResBlock(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = ResBlock(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = ResBlock(filters[3], filters[4])

        self.upnet4 = upconv(filters[4], filters[3])
        self.upnet3 = upconv(filters[3], filters[2])
        self.upnet2 = upconv(filters[2], filters[1])
        self.upnet1 = upconv(filters[1], filters[0])

        self.final = nn.Sequential(
            nn.Conv2d(filters[0], n_classes, kernel_size=1),
        )

    def forward(self, inputs):
        enc1 = self.conv1(inputs)
        pool1 = self.maxpool1(enc1)

        enc2 = self.conv2(pool1)
        pool2 = self.maxpool2(enc2)

        enc3 = self.conv3(pool2)
        pool3 = self.maxpool3(enc3)

        enc4 = self.conv4(pool3)
        pool4 = self.maxpool4(enc4)

        center = self.center(pool4)

        up4 = self.upnet4(enc4, center)
        up3 = self.upnet3(enc3, up4)
        up2 = self.upnet2(enc2, up3)
        up1 = self.upnet1(enc1, up2)

        out = self.final(up1)
        return out
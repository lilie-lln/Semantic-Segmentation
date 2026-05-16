import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet34Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3, stride=1) 
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        enc0 = self.conv1(x)  
        enc0 = self.bn1(enc0)
        enc0 = self.relu(enc0)
        enc0_pool = self.maxpool(enc0) 

        enc1 = self.layer1(enc0_pool) 
        enc2 = self.layer2(enc1)      
        enc3 = self.layer3(enc2)      
        enc4 = self.layer4(enc3)      

        return enc0, enc1, enc2, enc3, enc4
    
class UnetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)

class UnetDecoder(nn.Module):
    def __init__(self):
        super(UnetDecoder, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv4 = UnetConvBlock(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv3 = UnetConvBlock(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv2 = UnetConvBlock(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder_conv1 = UnetConvBlock(128, 64)

        self.final_upsample = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)

    def forward(self, enc0, enc1, enc2, enc3, enc4):
        up4 = self.upconv4(enc4)              
        dec4 = self.decoder_conv4(torch.cat([up4, enc3], dim=1))

        up3 = self.upconv3(dec4)              
        dec3 = self.decoder_conv3(torch.cat([up3, enc2], dim=1))

        up2 = self.upconv2(dec3)              
        dec2 = self.decoder_conv2(torch.cat([up2, enc1], dim=1))

        up1 = self.upconv1(dec2)              
        dec1 = self.decoder_conv1(torch.cat([up1, enc0], dim=1))

        out = self.final_upsample(dec1)       

        return out
    
class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResNet34_UNet, self).__init__()
        self.encoder = ResNet34Encoder(in_channels)
        self.decoder = UnetDecoder()

    def forward(self, x):
        enc0, enc1, enc2, enc3, enc4 = self.encoder(x)
        out = self.decoder(enc0, enc1, enc2, enc3, enc4)
        return out
    
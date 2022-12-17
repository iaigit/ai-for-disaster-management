import torch
import torch.nn as nn
import torch.nn.functional as F

class Fire(nn.Module):
    def __init__(self, input_channel, s1x1, e1x1, e3x3):
        super(Fire, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, s1x1, kernel_size=1)
        self.conv2 = nn.Conv2d(s1x1, e1x1, kernel_size=1)
        self.conv3 = nn.Conv2d(input_channel, e3x3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(e3x3, e3x3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)

        return x

class DeFire(nn.Module):
    def __init__(self, input_channel, s1x1, e1x1, e3x3, last_channel):
        super(DeFire, self).__init__()
        self.conv1 = ConvBlock(1, 1, 0, input_channel, s1x1)
        self.conv2 = ConvBlock(3, 1, 1, s1x1, e1x1)
        self.conv3 = ConvBlock(1, 1, 0, e1x1, e3x3)
        self.resample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv4 = ConvBlock(3, 1, 1, e3x3, last_channel)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resample(x)
        x = self.conv4(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, kernel_size, stride, padding, in_channels, out_channels, max_pool=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = max_pool

        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.max_pool:
            x = self.max_pool(x)
        return x


class AttSqueezeUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_1 = ConvBlock(kernel_size=7, stride=1, padding=3, in_channels=3, out_channels=64, max_pool=True)
        self.down_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Fire(64, 16, 64, 64),
            Fire(64, 16, 64, 64),
            Fire(64, 16, 64, 128),
        )
        self.down_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Fire(128, 32, 128, 128),
            Fire(128, 32, 128, 128),
            Fire(128, 32, 128, 128),
            Fire(128, 32, 128, 256),
        )
        self.down_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Fire(256, 16, 256, 256),
            ConvBlock(kernel_size=1, stride=1, padding=0, in_channels=256, out_channels=256),
            ConvBlock(kernel_size=3, stride=1, padding=1, in_channels=256, out_channels=512),
        )
        self.up_1 = DeFire(512, 32, 512, 32, 256)
        self.up_2 = nn.Sequential(
            ConvBlock(kernel_size = 3, stride=1, padding=1, in_channels=512, out_channels=256),
            DeFire(256, 16, 256, 16, 128),
        )
        self.up_3 = nn.Sequential(
            ConvBlock(kernel_size = 3, stride=1, padding=1, in_channels=256, out_channels=128),
            DeFire(128, 16, 128, 16, 64),
        )
        self.up_4 = nn.Sequential(
            ConvBlock(kernel_size = 3, stride=1, padding=1, in_channels=128, out_channels=64),
            DeFire(64, 16, 64, 16, 64),
        )
        self.last_layer = nn.Sequential(
            ConvBlock(kernel_size = 3, stride=1, padding=1, in_channels=64, out_channels=64),
            ConvBlock(kernel_size = 1, stride=1, padding=0, in_channels=64, out_channels=64),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
        )
    def forward(self, x):
        d_1 = self.down_1(x)
        print('d_1: ', d_1.shape)
        d_2 = self.down_2(d_1)
        print('d_2: ', d_2.shape)
        d_3 = self.down_3(d_2)
        print('d_3: ', d_3.shape)
        d_4 = self.down_4(d_3)
        print('d_4: ', d_4.shape)

        u_1 = self.up_1(d_4)
        print('u_1: ', u_1.shape)
        u_2 = self.up_2(torch.concat([u_1, d_3], dim=1))
        print('u_2: ', u_2.shape)
        u_3 = self.up_3(torch.concat([u_2, d_2], dim=1))
        print('u_3: ', u_3.shape)
        u_4 = self.up_4(torch.concat([u_3, d_1], dim=1))
        print('u_4: ', u_4.shape)
        last_layer = self.last_layer(u_4)
        print('last_layer: ', last_layer.shape)

        return last_layer


if __name__ == "__main__":
    # test
    model = AttSqueezeUnet()
    print(model)
    x = torch.randn(1, 3, 224, 224)
    print(x.shape)
    y = model(x)
    print(y.shape)
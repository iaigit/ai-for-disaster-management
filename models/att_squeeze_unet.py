import torch
import torch.nn as nn
import torch.nn.functional as F

#it's from implementation of SqueezeNet in Pytorch -> torchvision.models.squeezenet
class Fire(nn.Module):

    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class ATTSqueezeUnet(nn.Module):
    
        def __init__(
            self,
            num_classes: int = 1,
            num_channels: int = 3
        ) -> None:
            super(ATTSqueezeUnet, self).__init__()
            self.num_classes = num_classes
            self.num_channels = num_channels
            
            x01 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=1, padding=3)
            x01_act = nn.ReLU(inplace=True)
            x01_maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
            x02 = Fire(64, 16, 64, 64)
            x03 = Fire(128, 16, 64, 64)
            x04 = Fire(128, 32, 128, 128)

            x05 = nn.MaxPool2d(kernel_size=3, stride=2)
            x06 = Fire(256, 32, 128, 128)
            x07 = Fire(256, 48, 192, 192)
            x08 = Fire(384, 48, 192, 192)
            x09 = Fire(384, 64, 256, 256)
            x10 = nn.MaxPool2d(kernel_size=3, stride=2)
            x11 = Fire(512, 64, 256, 256)

            x12 = nn.Conv2d(512, 512, kernel_size=1)
            x12_act = nn.ReLU(inplace=True)
            x13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            x13_act = nn.ReLU(inplace=True)

            up1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
            up1_act = nn.ReLU(inplace=True)
            
from numpy import squeeze
import torch
import torch.nn as nn
from torchvision import models

def SqueezeNet(num_classes: int) -> nn.Module:
    squeezenet = models.squeezenet1_1(pretrained=False, num_classes=num_classes)

    return squeezenet

if __name__ == '__main__':
    squeezenet_model = SqueezeNet(100)
    print(squeezenet_model)
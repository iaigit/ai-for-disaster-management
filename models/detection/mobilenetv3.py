import torch
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v3_large
from torch import nn, optim
from torchvision import datasets, transforms


class ForestFireRecognition(nn.Module):
    def __init__(self):
        super().__init__()
        self.mnet = model = mobilenet_v3_large()
        self.mnet.classifier = nn.Sequential(
            nn.Hardswish(),
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, 2),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.mnet(x)
        return x


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class Predict(nn.Module):
    def __init__(self):
        super().__init__()
        self.label2cat = ["Fire", "No Fire"]
        self.model = ForestFireRecognition()
        self.model.load_state_dict(torch.load(
            "../../weights/detection/mobilenetv3_98.pth", map_location='cpu'))
        self.model.eval()

    def predict(self, img):
        img = test_transform(img)
        img = img[None, :]
        with torch.no_grad():
            out = self.model(img)
            pred = self.label2cat[out.argmax(1)[0]]
            pred_prob = torch.exp(out.max(1)[0]).item()
            # pred prob to 4 float
            pred_prob = round(pred_prob, 4)
        return pred, pred_prob

from torch import nn
from torch.nn import functional as F
import torch

class SoftDiceLoss(nn.Module):

    def __init__(self, smooth=1):
        """:arg smooth The parameter added to the denominator and numerator to prevent the division by zero"""

        super().__init__()
        self.smooth = smooth

    def forward(self, input, target):
        assert torch.max(target).item() <= 1, 'SoftDiceLoss() is only available for binary classification.'

        batch_size = input.size(0)
        probability = F.softmax(input, dim=1)

        # Convert probability and target to the shape [B, (C*H*W)]
        probability = probability.view(batch_size, -1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute((0, 3, 1, 2))
        target_one_hot = target_one_hot.contiguous().view(batch_size, -1)

        intersection = probability * target

        dsc = (2 * intersection.sum(dim=1) + self.smooth) / (
                target_one_hot.sum(dim=1) + probability.sum(dim=1) + self.smooth)
        loss = (1 - dsc).sum()

        return loss
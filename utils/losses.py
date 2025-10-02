import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
# from sklearn.utils import class_weight
from .lovasz_losses import lovasz_softmax

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target



class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):

        valid = (target != self.ignore_index)          # [N, H, W]  bool
        if valid.sum() == 0:                           # 整张图都是 ignore
            return torch.tensor(0., device=output.device)


        C = output.size(1)
        target = target.unsqueeze(1)                   # [N,1,H,W]
        target_onehot = torch.zeros_like(output)
        target_onehot.scatter_(1, target, 1)           # [N,C,H,W]
        target_onehot = target_onehot * valid.unsqueeze(1)


        prob = F.softmax(output, dim=1) * valid.unsqueeze(1)


        dims = (2, 3)                                  # 在 H,W 维度上求和
        inter = (prob * target_onehot).sum(dims)       # [N,C]
        union = prob.sum(dims) + target_onehot.sum(dims)
        dice  = (2. * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight
        self.dice = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.ce   = nn.CrossEntropyLoss(weight=None, reduction=reduction,
                                        ignore_index=ignore_index)

    def forward(self, output, target):

        if self.weight is not None:
            self.ce.weight = self.weight.to(output.device)
        CE_loss = self.ce(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss

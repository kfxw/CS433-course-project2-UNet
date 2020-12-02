import torch
import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):
    """Dice loss/coefficient of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [bs, c, h, w]
        target: A tensor of shape [bs, 1, h, w]
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss/Acc tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean', get_coefficient=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.get_coefficient = get_coefficient

    def forward(self, pred, gt):
        assert pred.shape[0] == gt.shape[0], "predict & target batch size don't match"

        if self.get_coefficient:            # compute loss with prob, evaluate with mask
            pred = (pred > 0.5).float()
        else:
            pred = torch.sigmoid(pred)
        
        pred = pred.contiguous().view(pred.shape[0], -1)
        gt = gt.contiguous().view(gt.shape[0], -1)
        num = torch.sum(torch.mul(pred, gt), dim=1) + self.smooth
        den = torch.sum(pred.pow(self.p) + gt.pow(self.p), dim=1) + self.smooth

        if self.get_coefficient:
            res = num / den
        else:
            res = 1 - num / den

        if self.reduction == 'mean':
            return res.mean()
        elif self.reduction == 'sum':
            return res.sum()
        elif self.reduction == 'none':
            return res
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

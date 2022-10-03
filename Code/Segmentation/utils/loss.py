import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class FocalTversky_BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ALPHA=0.3, BETA=0.7, GAMMA=0.75):
        super(FocalTversky_BCELoss, self).__init__()
        # ALPHA + BETA = 1
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.GAMMA = GAMMA

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs_ = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs_ = inputs_.view(-1)
        targets_ = targets.view(-1)

        #True Positives, False Positives & False Negatives
        TP = (inputs_ * targets_).sum()
        FP = ((1 - targets_) * inputs_).sum()
        FN = (targets_ * (1 - inputs_)).sum()

        Tversky = (TP + smooth) / (TP + self.ALPHA * FP + self.BETA * FN + smooth)
        BCE = nn.BCEWithLogitsLoss()
        FocalTversky = (1 - Tversky)**self.GAMMA + BCE(inputs, targets)

        return FocalTversky


class Mul_FocalTversky_Loss(nn.Module):
    def __init__(self, weight=None, ALPHA=0.3, BETA=0.7, GAMMA=0.75):
        super(Mul_FocalTversky_Loss, self).__init__()
        # ALPHA + BETA = 1
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.GAMMA = GAMMA

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.softmax(inputs, dim=1)
        mean_loss = 0
        for i in range(inputs.shape[1]):
            inputs1 = torch.sigmoid(inputs)
            inputs1 = inputs1[:, i, :, :]
            targets1 = targets[:, i, :, :]
            #flatten label and prediction tensors
            inputs1 = inputs1.contiguous().view(-1)
            targets1 = targets1.contiguous().view(-1)

            #True Positives, False Positives & False Negatives
            TP = (inputs1 * targets1).sum()
            FP = ((1 - targets1) * inputs1).sum()
            FN = (targets1 * (1 - inputs1)).sum()

            Tversky = (TP + smooth) / (TP + self.ALPHA * FP + self.BETA * FN +
                                       smooth)
            # BCE = nn.BCEWithLogitsLoss()
            FocalTversky = (1 - Tversky)**self.GAMMA  #+ BCE(inputs1, targets1)
            mean_loss += FocalTversky
        return mean_loss / inputs.shape[1]


class Multi_focal_loss(nn.Module):
    def __init__(self, ALPHA=0.3, BETA=0.7, GAMMA=0.75):
        super(Multi_focal_loss, self).__init__()
        # ALPHA + BETA = 1
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.GAMMA = GAMMA

    def forward(self, inputs, targets):
        total_loss = 0
        for i in range(inputs.shape[1]):
            # print(inputs.shape)
            sigmoid_p = torch.sigmoid(inputs[:, i, :, :])
            zeros = torch.zeros_like(sigmoid_p)
            pos_p_sub = torch.where(targets > zeros, targets - sigmoid_p,
                                    zeros)
            neg_p_sub = torch.where(targets > zeros, zeros, sigmoid_p)
            per_entry_cross_ent = -self.ALPHA * (
                pos_p_sub**self.GAMMA) * torch.log(
                    torch.clamp(sigmoid_p, 1e-8, 1.0)) - (1 - self.ALPHA) * (
                        neg_p_sub**self.GAMMA) * torch.log(
                            torch.clamp(1.0 - sigmoid_p, 1e-8, 1.0))
            total_loss += per_entry_cross_ent.sum()
        return total_loss / inputs.shape[1]


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1),
                                 -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]

        logits = F.log_softmax(logits, 1)
        logits = logits.gather(1, target)  # [NHW, 1]
        loss = -1 * logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class CrossEntropyFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.2, reduction='mean'):
        super(CrossEntropyFocalLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1),
                                 -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]

        pt = F.softmax(logits, 1)
        pt = pt.gather(1, target).view(-1)  # [NHW]
        log_gt = torch.log(pt)

        if self.alpha is not None:
            # alpha: [C]
            alpha = self.alpha.gather(0, target.view(-1))  # [NHW]
            log_gt = log_gt * alpha

        loss = -1 * (1 - pt)**self.gamma * log_gt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() +
                                                        targets.sum() + smooth)
        BCE = nn.BCEWithLogitsLoss()
        Dice_BCE = BCE(inputs, targets) + dice_loss

        return Dice_BCE


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() +
                                               smooth)
        return 1 - dice



class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ALPHA=0.8, BETA=2):
        super(FocalLoss, self).__init__()
        self.ALPHA = ALPHA
        self.BETA = BETA

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.ALPHA * (1 - BCE_EXP)**self.BETA * BCE

        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ALPHA=0.5, BETA=0.5):
        super(TverskyLoss, self).__init__()
        self.ALPHA = ALPHA
        self.BETA = BETA

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.ALPHA * FP + self.BETA * FN +
                                   smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self,
                 weight=None,
                 size_average=True,
                 ALPHA=0.3,
                 BETA=0.7,
                 GAMMA=0.75):
        super(FocalTverskyLoss, self).__init__()
        # ALPHA + BETA = 1
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.GAMMA = GAMMA

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.ALPHA * FP + self.BETA * FN +
                                   smooth)
        FocalTversky = (1 - Tversky)**self.GAMMA

        return FocalTversky


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


"""
In this PyTorch file, we provide implementations of our new loss function, ASL, 
that can serve as a drop-in replacement for standard loss functions (Cross-Entropy and Focal-Loss)

For the multi-label case (sigmoids), the two implementations are:
        class AsymmetricLoss(nn.Module)
        class AsymmetricLossOptimized(nn.Module)

The two losses are bit-accurate. However, AsymmetricLossOptimized() contains a more optimized (and complicated) 
way of implementing ASL, which minimizes memory allocations, gpu uploading, and favors inplace operations.

For the single-label case (softmax), the implementations is called:
        class ASLSingleLabel(nn.Module)

"""
class AsymmetricLoss(nn.Module):
    def __init__(self,
                 gamma_neg=4,
                 gamma_pos=1,
                 clip=0.05,
                 eps=1e-8,
                 disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''
    def __init__(self,
                 gamma_neg=4,
                 gamma_pos=1,
                 clip=0.05,
                 eps=1e-8,
                 disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets *
                       torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg, self.gamma_pos * self.targets +
                self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    def __init__(self,
                 gamma_pos=0,
                 gamma_neg=4,
                 eps: float = 0.1,
                 reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []  # prevent gpu repeated memory allocation
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target, reduction=None):
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        print(log_preds.shape)
        print(target.shape)
        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1,
            target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes.mul_(1 - self.eps).add_(self.eps /
                                                         num_classes)

        # loss calculation
        loss = -self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss
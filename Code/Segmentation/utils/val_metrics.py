import torch
from torch.autograd import Function


class val_DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def val_dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + val_DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class mul_val_DiceCoeff(Function):
    """Dice coeff for individual examples"""
    # def forward(self, input, target):
    #     self.save_for_backward(input, target)
    #     eps = 0.0001
    #     mean_t = 0
        
    #     # for i in range(input.shape[0]):
            
    #         # self.inter = torch.dot(input[i, :, :].contiguous().view(-1),
    #         #                        target[i, :, :].contiguous().view(-1))
    #         # self.union = torch.sum(input[i, :, :]) + torch.sum(target[i, :, :]) + eps
    #         # self.inter = torch.dot(input[i, :, :].view(-1),
    #         #                        target.view(-1))
    #         # self.union = torch.sum(input[i, :, :]) + torch.sum(target) + eps

    #         # mean_t += (2 * self.inter.float() + eps) / self.union.float()

        
    #     return mean_t / input.shape[0]
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def mul_val_dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + mul_val_DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class val_FocalLoss(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        inputs = input.view(-1)
        targets = target.view(-1)

        #True Positives, False Positives & False Negatives
        self.TP = torch.sum(torch.dot(inputs, targets))
        self.FP = torch.sum(torch.dot(1 - targets, inputs))
        self.FN = torch.sum(torch.dot(targets, 1 - inputs))

        Tversky = (self.TP + eps) / (self.TP + 0.3 * self.FP + 0.7 * self.FN +
                                     eps)
        t = (1 - Tversky)**0.75
        return t

    # This function has only a single output, so it gets only one gradient


def val_Focal_loss(input, target):
    """Focal_loss"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + val_FocalLoss().forward(c[0], c[1])

    return s / (i + 1)

import torch
import torch.nn as nn

THRESH = .01 

class GradClipMath(torch.autograd.Function):
    '''
    forward pass; do nothing
    backward pass: make sure losses are in safe range, drop to 0 otherwsie
    '''
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        o = torch.zeros_like(grad_x)

        #if outside thresh/NaN, drop to 0
        grad_x = torch.where(grad_x.abs()>THRESH, o, grad_x)
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x)
        return grad_x

class GradientClipLayer(nn.Module):
    def __init__(self):
        super(GradientClipLayer, self).__init__()

    def forward(self, x):
        return GradClipMath.apply(x)
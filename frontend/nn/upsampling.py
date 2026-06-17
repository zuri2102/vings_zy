import torch
import torch.nn.functional as F

def convex_up(data, mask):
    '''
    upsample/upscale, (learned, mask) weighted average of surrounding 3x3 
    NOT blurring
    '''
    B, H, W, Dim = data.shape
    data = data.permute(0, 3, 1, 2) #B, Dim, H, W
    mask = mask.view(B, 1, 9, 8, 8, H, W) #9 for 3x3
    weights = torch.softmax(mask, dim=2)

    data = F.unfold(data, [3,3], padding=1)
    data = data.view(B, Dim, 9, 1, 1, H, W)
    upscaled = torch.sum(weights * data, dim=2)
    upscaled = upscaled.reshape(B, 8*H, 8*W, Dim)

    return upscaled

def upsample_invdepth(invdepth, mask):
    '''
    frontend pipeline part, applying convex_up
    '''
    B, N, H, W = invdepth.shape
    invdepth = invdepth.view(B*N, H, W, 1)
    mask = mask.view(B*N, -1, H, W)
    return convex_up(invdepth, mask).view(B, N, 8*H, 8*W)
    

    



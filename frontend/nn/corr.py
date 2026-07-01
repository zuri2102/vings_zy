import torch
import torch.nn.functional as F
import droid_backends as droid_backends_nerfslam

class CorrLayer(torch.autograd.Function):
    '''
    droid: multiplies feature maps quickly to get final correlation maps
    checks correlation within a radius
    '''
    @staticmethod
    def forward(ctx, fmap_i, fmap_j, coords, r):
        ctx.r = r
        ctx.save_for_backward(fmap_i, fmap_j, coords)
        corr, = droid_backends_nerfslam.altcorr_forward(fmap_i, fmap_j, coords, ctx.r)
        return corr
    
    @staticmethod
    def backward(ctx, grad_corr):
        fmap_i, fmap_j, coords = ctx.saved_tensors
        grad_corr = grad_corr.contiguous()
        fmap_i_grad, fmap_j_grad, coords_grad = \
            droid_backends_nerfslam.altcorr_backward(fmap_i, fmap_j, coords, grad_corr, ctx.r)
        return fmap_i_grad, fmap_j_grad, coords_grad, None

class AltCorrBlock:
    '''
    better bc computes only when necessary--saves raw, checks within radius
    
    
    @arg fmaps : takes in 2, i and j, before and after

    @arg num_levels : indicates how much downsize by
    '''
    def __init__(self, fmaps, num_levels=4, radius = 3): #num_levels -> downsample by 2^3
        self.num_levels = num_levels
        self.radius = radius
        B, N, C, H, W = fmaps.shape
        fmaps = fmaps.view(B*N, C, H, W) / 4 #4 bc later on, matrix multiplication of 2 maps, so /16
        self.pyramid = [] #putting downsampled levels
        for i in range(self.num_levels):
            sz = (B, N, H//2**i, W//2**i, C)
            fmap_lvl = fmaps.permute(0, 2, 3, 1).contiguous()
            self.pyramid.append(fmap_lvl.view(*sz))
            fmaps = F.avg_pool2d(fmaps, 2, stride=2) #blur and shrink
    
    def corr_fn(self, coords, i, j):
        """
        Corr math

        i is before index, j is after index

        each level holds correlation of before and after, before always being high res, and after being increasingly less blurry
        """
        B, N, H, W, S, _ = coords.shape #S = sample?
        coords = coords.permute(0, 1, 4, 2, 3, 5)

        corr_list = []
        for k in range(self.num_levels):
            r = self.radius
            fmap_i = self.pyramid[0][:, i] #crisp vs blurry and blurrier
            fmap_j = self.pyramid[k][:, j]

            coords_k = (coords / (2**k)).reshape(B*N, S, H, W, 2).contiguous()
            fmap_i = fmap_i.reshape((B*N,) + fmap_i.shape[2:])
            fmap_j = fmap_j.reshape((B*N,) + fmap_i.shape[2:])

            corr = CorrLayer.apply(fmap_i.float(), fmap_j.float(), coords_k, self.radius) #get correlation 
            corr = corr.view(B, N, S, -1, H, W).permute(0, 1, 3, 4, 5, 2)
            corr_list.append(corr)

        corr = torch.cat(corr_list, dim=2)
        return corr
    
    def __call__(self, coords, i, j):
        """
        Runs correlation
        """
        squeeze_output = False
        if len(coords.shape) == 5: #if B, N, H, W, 2, aka no S->only one hypotheses
            coords = coords.unsqueeze(dim=-2)
            squeeze_output = True

        corr = self.corr_fn(coords, i, j)
        
        if squeeze_output:
            corr = corr.squeeze(dim=-1)

        return corr.contiguous()
        
        


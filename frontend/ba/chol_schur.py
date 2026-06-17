import torch

EP = 0.1
LM = 0.0001

class CholeskySolver(torch.autograd.Function):
    '''
    LL^T = H, SOLVE Hx = b,
    '''
    @staticmethod
    def forward(ctx, H, b):
        L, info = torch.linalg.cholesky_ex(H) #ex is faster

        #if H is not hermitian positive definite, info > 0. MASK OUT BAD:
        succ = (info == 0)
        mask = succ.float()[...,None, None] 

        x = torch.cholesky_solve(b, L) * mask
        ctx.save_for_backward(L, x, mask)
        return x   
        
    @staticmethod
    def backward(ctx, grad_x):
        L, x, mask = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, L) * mask #solve H * dz = grad_x
        dH = -torch.matmul(dz, x.transpose(-1,-2)) #dH = -dz * x^T
        dH = 0.5 * (dH + dH.transpose(-1, -2)) #force symmetry for next cholesky call

        return dH, dz
    
def norm_solve(H, b, ep=EP, lm=LM):
    '''
    normal eq, Hx = b
    '''
    B, N, _, D, _ = H.shape
    
    H = H.permute(0,1,3,2,4)
    H = H.reshape(B, N*D, N*D)
    b = b.reshape(B, N*D, 1)
    
    d = torch.diagonal(H, dim1=-2, dim2=-1)
    d_new = d + ep + lm * d
    H = H.diagonal_scatter(d_new, dim1=-2, dim2=-1)

    x = CholeskySolver.apply(H,b)
    return x.reshape(B, N, D)

def schur_solve(H, E, C, v, w, ep=EP, lm=LM, sless=False):
    '''
    SCHUR COMPLEMENT

    sless -> pose only, DO NOT update map
    
    S = H - E C^(-1)E^T
    H = pose hessian
    v = pose error
    dx = pose adjustment

    C = point hessian
    w = point error
    dz = point adjustment

    E = camera/point interaction jacobian
    Q = 1/C

    (1) solve for cam pose: S dx = v - E C^(-1)w
    (2) solve for inverse depth update: dz = C^(-1) * (w - E^T dx)
    '''
    B, P, M, D, HW = E.shape
    H = H.permute(0,1,3,2,4).reshape(B, P*D, P*D)
    E = E.permute(0,1,3,2,4).reshape(B, P*D, M*HW)
    Q = (1.0 / (C + 1e-7)).view(B, M*HW, 1) #epsilon to prevent division by 0

    #damping
    d = torch.diagonal(H, dim1=-2, dim2=-1)
    d_new = d + ep + lm * d
    H = H.diagonal_scatter(d_new, dim1=-2, dim2=-1)

    v = v.reshape(B, P*D, 1)
    w = w.reshape(B, M*HW, 1)

    Et = E.transpose(1,2)
    S = H - torch.matmul(E, Q*Et) #S = H - E C^(-1)E^T
    v = v - torch.matmul(E, Q*w)

    #force symmetry before cholesky

    #(1) S dx = v - E C^(-1)w
    dx = CholeskySolver.apply(S, v)
    if sless:
        return dx.reshape(B, P, D)

    #(2) dz = C^(-1) * (w - E^T dx)
    dz = Q * (w - Et @ dx)    
    dx = dx.reshape(B, P, D)
    dz = dz.reshape(B, M, HW)

    return dx, dz
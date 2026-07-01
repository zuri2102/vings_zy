import torch
import lietorch
from . import proj_math as props
from .chol_schur import norm_solve, schur_solve
from torch_scatter import scatter_sum

WEIGHT_FACTOR = 0.001

def safe_scatter_sum_mat(A, i, j, n, m):
    v = (i >= 0) & (j >= 0) & (i < n) & (j < m)
    return scatter_sum(A[:, v], i[v]*m + j[v], dim = 1, dim_size = n*m)

def safe_scatter_sum_vec(b, i, n):
    v = (i > 0) & (i < n)
    return scatter_sum(b[:, v], i[v], dim=1, dim_size=n)


def pose_update(poses, dx, i):
    i = i.to(device=dx.device)
    return poses.retr(scatter_sum(dx, i, dim=1, dim_size=poses.shape[1]))

    
def invdepth_update(invdepth, dz, i):
    i = i.to(device=dz.device)
    return invdepth + scatter_sum(dz, i, dim=1, dim_size=invdepth.shape[1])


def BA(target, weight, damp_factor, poses, invdepth, intrinsics, i, j, frame_offset=1, num_cams=1):
    '''
    full bundle adjustment
    minimizing: cost = Σ [ w * (Target - Projection(P, d))² ]

    '''
    B, P, height, width, = invdepth.shape #B = batch, P = poses/keyframes
    N = i.shape[0] #number i/j's
    D = poses.manifold_dim

    #jacobians and residuals
    coords, valid, (J_i, J_j, J_z) = props.full_proj_transform(poses, invdepth, intrinsics, i, j, jacobian=True)

    r = (target - coords).view(B, N, -1, 1) #adjusting poses and depths until difference between GRU(?) and math is minimal
    w = WEIGHT_FACTOR * (valid * weight).view(B, N, -1, 1) 

    #construct lin systems of eqs
    J_i = J_i.reshape(B, N, -1, D) #(B, N, 8192, 6), 8192 observations for 6 cam vars
    J_j = J_j.reshape(B, N, -1, D)
    J_z = J_z.reshape(B, N, height*width, -1) #(B, N, 4096, 2), 4096 pixels, each with 2D error

    JiT_w = (w * J_i).transpose(2,3)
    JjT_w = (w * J_j).transpose(2,3)

    #H ~ J_T * w * J 
    H_ii = torch.matmul(JiT_w, J_i)
    H_jj = torch.matmul(JjT_w, J_j)
    H_ij = torch.matmul(JiT_w, J_j)
    H_ji = torch.matmul(JjT_w, J_i)

    #v = J^T * w * r -> gradient, 6D vector to move to reduce error, CAM
    vi = torch.matmul(JiT_w, r).squeeze(-1)
    vj = torch.matmul(JjT_w, r).squeeze(-1)

    #E = J^T * w * J -> coupling pose and depth
    E_i = (JiT_w.view(B, N, D, height*width, -1) * J_z[:, :, None]).sum(dim=-1)
    E_j = (JjT_w.view(B, N, D, height*width, -1) * J_z[:, :, None]).sum(dim=-1)
    
    w = w.view(B, N, height*width, -1)
    r = r.view(B, N, height*width, -1)

    wk = torch.sum(w*r*J_z, dim=-1) #PIXEL DEPTH gradient
    Ck = torch.sum(w*J_z*J_z, dim=-1) #H_zz

    kx, kk = torch.unique(i, return_inverse=True) #IDing (kx = val, kk = index)
    M = kx.shape[0] # number of unique items

    #"reindexing" for specific frames
    P = P // num_cams - frame_offset
    i = i // num_cams - frame_offset
    j = j // num_cams - frame_offset

    #combining & sorting to frames
    H = (
        safe_scatter_sum_mat(H_ii, i, i, P, P) +
        safe_scatter_sum_mat(H_ij, i, j, P, P) +
        safe_scatter_sum_mat(H_ji, j, i, P, P) +
        safe_scatter_sum_mat(H_jj, j, j, P, P)
    )

    E = (
        safe_scatter_sum_mat(E_i, i, kk, P, M) + 
        safe_scatter_sum_mat(E_j, j, kk, P, M)
    )

    v = (
        safe_scatter_sum_vec(vi, i, P) + 
        safe_scatter_sum_vec(vj, j, P)
    )

    C = safe_scatter_sum_vec(Ck, kk, M)
    W = safe_scatter_sum_vec(wk, kk, M)

    C = C + damp_factor.view(*C.shape) + 1e-7 #epsilon so no 0
    
    H = H.view(B, P, P, D, D)
    E = E.view(B, P, M, D, height*width)

    # solve system 
    dx, dz = schur_solve(H, E, C, v, W)

    # apply to cam trajectories (poses) and map (invdepth)
    poses = pose_update(poses, dx, torch.arange(P) + frame_offset)
    invdepth = invdepth_update(invdepth, dz.view(B, -1, height, width), kx)

    invdepth = torch.where(invdepth > 10, torch.zeros_like(invdepth), invdepth) #if invdepth is too big, set to 0 (aka infinite depth)
    invdepth = invdepth.clamp(min=0)

    return poses, invdepth

def MO_BA(target, weight, damp_factor, poses, invdepth, intrinsics, i, j, frame_offset=1, num_cams=1):
    '''
    motion only, no updating map (treating map as static/100% correct), move cam until matches
    '''
    B, P, height, width, = invdepth.shape #B = batch, P = poses/keyframes
    N = i.shape[0] #number i/j's
    D = poses.manifold_dim

    #jacobians and residuals
    coords, valid, (J_i, J_j, J_z) = props.full_proj_transform(poses, invdepth, intrinsics, i, j, jacobian=True)

    r = (target - coords).view(B, N, -1, 1) #adjusting poses and depths until difference between GRU(?) and math is minimal
    w = WEIGHT_FACTOR * (valid * weight).view(B, N, -1, 1) 

    #construct lin systems of eqs
    J_i = J_i.reshape(B, N, -1, D) #(B, N, 8192, 6), 8192 observations for 6 cam vars
    J_j = J_j.reshape(B, N, -1, D)
    # J_z = J_z.reshape(B, N, height*width, -1) #(B, N, 4096, 2), 4096 pixels, each with 2D error

    JiT_w = (w * J_i).transpose(2,3)
    JjT_w = (w * J_j).transpose(2,3)

    #H ~ J_T * w * J 
    H_ii = torch.matmul(JiT_w, J_i)
    H_jj = torch.matmul(JjT_w, J_j)
    H_ij = torch.matmul(JiT_w, J_j)
    H_ji = torch.matmul(JjT_w, J_i)

    #v = J^T * w * r -> gradient, 6D vector to move to reduce error, CAM
    vi = torch.matmul(JiT_w, r).squeeze(-1)
    vj = torch.matmul(JjT_w, r).squeeze(-1)

    # #E = J^T * w * J -> coupling pose and depth
    # E_i = (JiT_w.view(B, N, D, height*width, -1) * J_z[:, :, None]).sum(dim=-1)
    # E_j = (JjT_w.view(B, N, D, height*width, -1) * J_z[:, :, None]).sum(dim=-1)
    
    # w = w.view(B, N, height*width, -1)
    # r = r.view(B, N, height*width, -1)

    # wk = torch.sum(w*r*J_z, dim=-1) #PIXEL DEPTH gradient
    # Ck = torch.sum(w*J_z*J_z, dim=-1) #H_zz

    # kx, kk = torch.unique(i, return_inverse=True) #IDing (kx = val, kk = index)
    # M = kx.shape[0] # number of unique items

    #"reindexing" for specific frames
    P = P // num_cams - frame_offset
    i = i // num_cams - frame_offset
    j = j // num_cams - frame_offset

    #combining & sorting to frames
    H = (
        safe_scatter_sum_mat(H_ii, i, i, P, P) +
        safe_scatter_sum_mat(H_ij, i, j, P, P) +
        safe_scatter_sum_mat(H_ji, j, i, P, P) +
        safe_scatter_sum_mat(H_jj, j, j, P, P)
    )

    # E = (
    #     safe_scatter_sum_vec(E_i, i, kk, P, M) + 
    #     safe_scatter_sum_vec(E_j, j, kk, P, M)
    # )

    v = {
        safe_scatter_sum_vec(vi, i, P) + 
        safe_scatter_sum_vec(vj, j, P)
    }

    # C = safe_scatter_sum_vec(Ck, kk, M)
    # W = safe_scatter_sum_vec(wk, kk, M)

    # C = C + damp_factor.view(*C.shape) + 1e-7 #epsilon so no 0
    
    H = H.view(B, P, P, D, D)
    # E = E.view(B, P, M, D, height*width)

    # solve system 
    dx = norm_solve(H, v)

    # apply to cam trajectories (poses) and map (invdepth)
    poses = pose_update(poses, dx, torch.arange(P) + frame_offset)
    # invdepth = invdepth_update(invdepth, dz.view(B, -1, height, width), kx)

    # invdepth = torch.where(invdepth > 10, torch.zeros_like(invdepth), invdepth) #if invdepth is too big, set to 0 (aka infinite depth)
    # invdepth = invdepth.clamp(min=0)

    return poses










    

    


























    
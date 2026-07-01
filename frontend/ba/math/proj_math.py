import torch
from lietorch import SE3, Sim3

MIN_DEPTH = 0.2 #TBD
STEREO_EXTRINSICS = [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] #tx,ty,tz,qx,qy,qz,qw; CURRENT: second cam is 0.1m to left of first 

def extract_intrinsics(intrinsics):
    fx = intrinsics[..., 0] #focal lengths
    fy = intrinsics[..., 1]
    cx = intrinsics[..., 2] #centerpoint
    cy = intrinsics[..., 3]

    #[B] -> [B, 1, 1], lag = Fast two dims for height and width
    fx = fx.view(*fx.shape, 1, 1)
    fy = fy.view(*fy.shape, 1, 1)
    cx = cx.view(*cx.shape, 1, 1)
    cy = cy.view(*cy.shape, 1, 1)

    return fx, fy, cx, cy

def coords_grid(h, w, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(h).to(**kwargs).float(),
        torch.arange(w).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)


def iproj(invdepth, intrinsics, jacobian=False):
    '''
    inverse projection, image to 3d point cloud 
    JACOBIAN TRUE ONLY FOR BACKPROP
    '''
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    height, width = invdepth.shape[2:]

    #initialize 2D
    y, x = torch.meshgrid(
        torch.arange(height).to(invdepth.device),
        torch.arange(width).to(invdepth.device)
    )

    #light rays and normalizing
    X = (x.float() - cx) / fx
    Y = (y.float() - cy) / fy
    onesunit = torch.ones_like(invdepth) #Z
    points = torch.stack([X, Y, onesunit, invdepth], dim=-1) #invdepth is how far along the ray object is, dim = -1 for speed

    if jacobian:
        #initialize [0,...,1.0]; 1.0->linear relationship with last element
        J = torch.zeros_like(points)
        J[...,-1] = 1.0
        return points, J
    
    return points, None

def proj(points3D, intrinsics, jacobian=False, geom3d=False, min_depth=MIN_DEPTH): 
    '''
    3d cloud onto 2d image OR to build 3d map
    '''
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, invdepth = points3D.unbind(dim=-1)

    #safeguard: if dangerously small, reset to 1.0 so next step is not divison by 0
    Z = torch.where(Z < 0.5*min_depth, torch.ones_like(Z), Z)

    #POST MOVEMENT, must update invdepth given new Z
    new_invdepth = 1.0 / Z
    x = fx * new_invdepth * X + cx
    y = fy * new_invdepth * Y + cy
    if geom3d:
        coords = torch.stack([x,y, new_invdepth*invdepth], dim=-1) #WHY NEW * OLD???????????????????????????????????????????????????
    else:
        coords = torch.stack([x,y], dim=-1)
    
    if jacobian:
        zero = torch.zeros_like(new_invdepth)

        #derivatives in terms of X, Y, Z, and invdepth
        J = torch.stack([fx*new_invdepth, zero, -fx*X*new_invdepth*new_invdepth, zero, #how x changes(left/right)
                         zero, fy*new_invdepth, -fy*Y*new_invdepth*new_invdepth, zero, #how y changes (up/down)
                        #  zero, zero, -invdepth*new_invdepth*new_invdepth, new_invdepth #how depth changes, OPTIONAL if 2d output
                        ], dim=-1)
        #reshaping flat derivatives vector(8) to a matrix(2x4)
        B, N, H, W = new_invdepth.shape
        J = J.view(B, N, H, W, 2, 4)
        
        return coords, J

    return coords, None

def move_in_pc(Tij, Xi, jacobian=False):
    '''
    apply transformation i->j
    '''
    Xj = Tij[:, :, None, None] * Xi
    if jacobian:
        X, Y, Z, invdepth = Xj.unbind(dim=-1)
        zero = torch.zeros_like(invdepth)
        B, N, H, W = invdepth.shape

        #IF motion is ONLY rotation and translation
        if isinstance(Tij, SE3):
            J = torch.stack([invdepth, zero, zero, zero, Z, -Y,
                             zero, invdepth, zero, -Z, zero, X,
                             zero, zero, invdepth, Y, -X, zero,
                             zero, zero, zero, zero, zero, zero], dim=-1)
            J = J.view(B, N, H, W, 4, 6) #reformat
        
        #IF motion is rotation, translation, AND scale
        elif isinstance(Tij, Sim3):
            J = torch.stack([invdepth, zero, zero, zero, Z, -Y, X,
                             zero, invdepth, zero, -Z, zero, X, Y,
                             zero, zero, invdepth, Y, -X, zero, Z,
                             zero, zero, zero, zero, zero, zero, zero], dim=-1)
            J = J.view(B, N, H, W, 4, 7) #reformat
        
        return Xj, J
    return Xj, None

def full_proj_transform(poses, invdepth, intrinsics, i, j, stereo_extrinsics=STEREO_EXTRINSICS, body_T = None, jacobian=False, geom3d=False, min_depth=MIN_DEPTH): 
    '''
    transforming i->j
    '''

    #INVERSE: get depth maps for every clip in batch for source frame 
    Xi, J_invd = iproj(invdepth[:, i], intrinsics[:, i], jacobian=jacobian)

    #RELATIVE POSE; pose of cam j * inverse of cam i = delta <-transformation matrix of i->j
    Tij = poses[:, j] * poses[:, i].inv() 

    if i == j: #STEREO PAIR; shortcut, no movement so default to gap between cams
        Tij.data[:, i] = torch.as_tensor(stereo_extrinsics, device=poses.device)
    
    #MOVE
    Xj, J_move = move_in_pc(Tij, Xi, jacobian=jacobian)

    #PROJECT BACK
    xj_2D, J_proj = proj(Xj, intrinsics[:, j], jacobian=jacobian, geom3d=geom3d)

    #omit too close
    valid = ((Xj[...,2] > min_depth) & (Xi[...,2] > min_depth)).float()
    valid = valid.unsqueeze(-1) #adjust dims

    if jacobian:
        J_j = torch.matmul(J_proj, J_move) #target jacobian
        J_i = -Tij[:, :, None, None, None].adjT(J_j) #mirrored effect

        #account for body
        if body_T is not None:
            body_T = SE3(body_T)
            J_i = body_T[None, None, None, None, None].adjT(J_i)
            J_j = body_T[None, None, None, None, None].adjT(J_j)
        
        #flip for correction
        J_i *= -1.0
        J_j *= -1.0

        #reorder for GTSAM (x,y,z,wx,wy,wz) -> (wx, wy, wz, x, y, z)
        J_i = J_i[..., [3,4,5,0,1,2]]
        J_j = J_j[..., [3,4,5,0,1,2]]

        #link depth to pixel movement
        J_invd = Tij[:, :, None, None] * J_invd
        J_invd = torch.matmul(J_proj, J_invd.unsqueeze(-1)) 

        return xj_2D, valid, (J_i, J_j, J_invd)

    return xj_2D, valid, (None, None, None)

def predict_optical_flow(poses, invdepth, intrinsics, i, j):
    '''
    flow = target pos - initial pos
    '''
    height, width = invdepth.shape[2:]

    #initialize 2D
    y, x = torch.meshgrid(
        torch.arange(height).to(invdepth.device),
        torch.arange(width).to(invdepth.device)
    )

    coords_i = torch.stack([x,y], dim=-1)
    coords_j, valid, _ = full_proj_transform(poses, invdepth, intrinsics, i, j)

    return coords_j[..., :2] - coords_i, valid 




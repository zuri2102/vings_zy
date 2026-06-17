import torch

class StateBuffer():
    '''
    owns memories, groups things, etc
    init: scalars/flags
    allocate: tensors, call on first frame
    '''
    def __init__(self, device='cpu', buffer=128, dsf=8, stereo=False):
        """
        device: 'cpu' or 'cuda'
        buffer: max keyframes holding
        dsf: downsampling factor
        stereo: false for monocular
        """
        self.device = device
        self.buffer = buffer
        self.dsf    = dsf
        self.stereo = stereo
        
        #tracking indices, k=raw, kf=keyframe
        self.kf_idx      = 0 #index ptr to next empty slot in ring buffer
        self.last_kf_idx = 0 #most recent ACCEPTED keyframe index
        self.last_k      = None #most recent RAW frame index
 
        self.kf_idx_to_f_idx = {} #keyframe->raw dict
        self.f_idx_to_kf_idx = {} #raw->keyframe dict
 
        #state flags
        self.is_initialized = False
        self.global_ba      = False
        self.stop           = False
        self.compute_covariances = True
 
        # keyframe selection
        self.motion_filter_thresh = 2.4   # min mean flow [px] to accept frame
        self.keyframe_thresh      = 4.0   # min BA-distance to keep keyframe
 
        # graph
        self.max_age          = 25    # retire edges older than this
        self.max_factors      = 48    # max active edges 
        self.keyframe_warmup  = 8     # frames needed for first BA
        self.kf_init_count    = 8     # BA iters during initialization
 
        # optimization window
        self.frontend_window  = 25    # keyframes in BA window that are adjusted
        self.frontend_radius  = 2     # make edges between keyframes with this dist (aka i connects i-1 and i-2)
        self.frontend_nms     = 1     # NMS suppression radius for proximity/redundant edges
        self.frontend_thresh  = 16.0  # max flow distance to add proximity edge, larger does not count as edge
        self.beta             = 0.3   # rotation/translation blend weight, i.e. how much sharp turn vs long straight movement weighs

        # BA iterations
        self.iters1 = 4   # GRU+BA iters before keyframe decision
        self.iters2 = 2   # GRU+BA iters after keyframe accepted (refining)
 
        # uncertainty priors (used when IMU is added later)
        self.translation_sigma = 0.01   # [m]
        self.rotation_sigma    = 0.01   # [rad]
        self.sigma_idepth      = 0.1    # [1/m]
 
        # correlation impl: "volume" (fast, more memory) or "alt" (slower, less memory)
        self.corr_impl = "volume"
 
        self.ht = None   # feature height = H // dsf
        self.wd = None   # feature width  = W // dsf
        self.coords0 = None  # static pixel grid (ht, wd, 2), never changes
 
    def allocate(self, image_size, coords_grid_fn, initial_pose=None):
        H, W = image_size
        h, w = H // self.dsf, W // self.dsf
 
        self.ht = h
        self.wd = w
        self.coords0 = coords_grid_fn(h, w, device=self.device)
 
        # initial pose — identity by default
        if initial_pose is None:
            initial_pose = torch.tensor(
                [0., 0., 0., 0., 0., 0., 1.],  # [tx,ty,tz, qx,qy,qz,qw]
                dtype=torch.float, device=self.device)
 
        cameras = 2 if self.stereo else 1
 
        # raw sensor data, do NOT get modified
        self.cam0_timestamps = torch.zeros(
            self.buffer, dtype=torch.float, device=self.device)
 
        self.cam0_images = torch.zeros(
            self.buffer, 3, H, W, dtype=torch.uint8, device=self.device)
 
        self.cam0_intrinsics = torch.zeros(
            self.buffer, 4, dtype=torch.float, device=self.device)
        # [fu/dsf, fv/dsf, cu/dsf, cv/dsf] — already divided by dsf at write time. focal lengths and principle points
 
        # initialized to identity, updated by BA on every iteration
        # poses: world-to-cam SE3 as quaternion [tx,ty,tz,qx,qy,qz,qw]
        self.cam0_T_world = torch.zeros(
            self.buffer, 7, dtype=torch.float, device=self.device)
        self.cam0_T_world[:] = initial_pose   # initialize all slots to starting pose
 
        # inverse depths at feature resolution: 1/depth [1/m]
        # initialized to 1.0 (1 meter), updated by BA
        self.cam0_idepths = torch.ones(
            self.buffer, h, w, dtype=torch.float, device=self.device)
 
        # depth uncertainty from BA information matrix (Eq. 9 in paper)
        self.cam0_idepths_cov = torch.ones(
            self.buffer, h, w, dtype=torch.float, device=self.device)
        self.cam0_depths_cov  = torch.ones(
            self.buffer, h, w, dtype=torch.float, device=self.device)
 
        # upsampled (after BA), used for mapping
        self.cam0_idepths_up    = torch.zeros(
            self.buffer, H, W, dtype=torch.float, device=self.device)
        self.cam0_depths_cov_up = torch.ones(
            self.buffer, H, W, dtype=torch.float, device=self.device)
 
        # written once by feature_net/context_net, never modified
        # NOTE: torch.float for CPU (half unsupported); change to torch.half for GPU
        self.features_imgs = torch.zeros( # feature maps, used for corr 
            self.buffer, cameras, 128, h, w,
            dtype=torch.float, device=self.device)
        self.contexts_imgs = torch.zeros( #tanh(ctx[:128]) GRU hidden state init
            self.buffer, cameras, 128, h, w,
            dtype=torch.float, device=self.device)
        self.cst_contexts_imgs = torch.zeros( #relu(ctx[128:]) constant GRU input (inp)
            self.buffer, cameras, 128, h, w,
            dtype=torch.float, device=self.device)
 
        #track how pixels shift in space from one frame view to another
        self.correlation_volumes = None
 
        # shape when non-empty: (1, E, 128, h, w)
        self.gru_hidden_states = None
 
        # shape when non-empty: (1, E, 128, h, w)
        self.gru_contexts_input = None
 
        # GRU flow estimate u*_ij, target for BA (updated each GRU step)
        # shape: (1, E, h, w, 2)
        self.gru_estimated_flow = torch.zeros(
            1, 0, h, w, 2, dtype=torch.float, device=self.device)
 
        # GRU flow weight w_ij, confidence, diagonal of Σ_ij (updated each GRU step)
        # shape: (1, E, h, w, 2)
        self.gru_estimated_flow_weight = torch.zeros(
            1, 0, h, w, 2, dtype=torch.float, device=self.device)
 
        # damping factor, from FrameSummarizer(?)
        # shape: (buffer, h, w)
        self.damping = 1e-6 * torch.ones(
            self.buffer, h, w, dtype=torch.float, device=self.device)
 
        # i[e]: source keyframe index of edge e
        # j[e]: target keyframe index of edge e
        # age[e]: how many BA iterations this edge has survived, remove once too old
        self.i  = torch.zeros(0, dtype=torch.long, device=self.device)
        self.j  = torch.zeros(0, dtype=torch.long, device=self.device)
        self.age = torch.zeros(0, dtype=torch.long, device=self.device)
 
        # edges too old kept for history influence
        self.i_inactive = torch.zeros(0, dtype=torch.long, device=self.device)
        self.j_inactive = torch.zeros(0, dtype=torch.long, device=self.device)
 
        # flow/weight stored for inactive edges 
        self.gru_estimated_flow_inactive = torch.zeros(
            1, 0, h, w, 2, dtype=torch.float, device=self.device)
        self.gru_estimated_flow_weight_inactive = torch.zeros(
            1, 0, h, w, 2, dtype=torch.float, device=self.device)
 
        # True at slot k means keyframe k has been updated since last viz
        # read to decide which keyframes to send to mapper (only) ones that have changed
        self.viz_idx = torch.zeros(
            self.buffer, dtype=torch.bool, device=self.device)
        
        
    @property
    def num_edges(self):
        return self.i.shape[0]
 
    @property
    def is_allocated(self):
        return self.coords0 is not None
 
    def active_keyframes(self):
        return torch.unique(self.i)
 
    def poses(self):
        import lietorch
        return lietorch.SE3(self.cam0_T_world[None])  # (1, buffer, 7)
 
    def idepths(self):
        return self.cam0_idepths[None]                # (1, buffer, 

import torch
import torch.nn as nn
from bundle_adj import safe_scatter_sum_mat
from state_buffer import StateBuffer
from update import UpdateModule
from extractor import BasicEncoder

class VIOFrontend(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.cfg = self.cfg
        self.device = self.device

        self.state = StateBuffer(device=device,
                                 buffer=cfg['frontend'].get('buffer', 128),
                                 dsf=8,
                                 stereo=False)
        
        #override anything necessary from state

        #build networks
        self.feature_net = BasicEncoder(output_dim=128, norm_fn='instance') #NOTE: is instance best??
        self.context_net = BasicEncoder(output_dim=256, norm_fn='none')
        self.update_net = UpdateModule()

        #load weights
        weights = self.load_weights(cfg['frontend']['weight']) #change depending on load method
        self.load_state_dict(weights)
        self.to(device)
        self.eval()

        #uncertainty priors
        self.translation_sigma = torch.tensor(0.01, device=self.device) # standard deviation of translation [m]
        self.rotation_sigma = torch.tensor(0.01, device=self.device) # standard deviation of rotation [rad]
        # TODO: What does this mean??? -> given that the values are much larger than 1.0... we should increase this much more...
        self.sigma_idepth = torch.tensor(0.1, device=self.device) # standard deviation of depth [m] (or inverse depth?) [1/m], we don't know the scale anyway...


    
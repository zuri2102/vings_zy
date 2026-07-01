'''
loop start 

rgb frame -> motion filter:
- run extractor, for feature map at low res: fnet
- put into corr, calculate corr between this frame and prev feature maps
- decide whether or not to toss (thresholding, if too small of a difference)--------------------------------------> if drop, next iteration

want to extract featrues and learn about scene now: cnet
cnet hidden state + feature desciription -> state buffer, now contains the new keyframe's information

graph:
- find keyframes that saw the same part of the scene (by camera distance)
- build correlation volumes for each new edge

if IMU available:
- integrate IMU measurements since last keyframe -> starting pose guess

update loop (runs multiple iterations):
- reproject pixels using current pose guesses
- sample correlation volumes at reprojection locations
- GRU: compare reprojection vs reality -> output delta + weight
- BA: solve for pose + depth corrections that explain all deltas across all edges simultaneously
- update poses + depths in state buffer

loop end
'''

import torch
import torch.nn as nn
from ba.bundle_adj import safe_scatter_sum_mat
from state_buffer import StateBuffer
from nn.update.update import UpdateModule
from nn.extractor import BasicEncoder

class VIOFrontend(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.cfg = cfg
        self.device = device

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
        weights = self.load_weights(cfg['frontend']['weight']) #change depending on load methodNA
        self.load_state_dict(weights)
        self.to(device)
        self.eval()

        #uncertainty priors
        self.translation_sigma = torch.tensor(0.01, device=self.device) # standard deviation of translation [m]
        self.rotation_sigma = torch.tensor(0.01, device=self.device) # standard deviation of rotation [rad]
        # TODO: What does this mean??? -> given that the values are much larger than 1.0... we should increase this much more...
        self.sigma_idepth = torch.tensor(0.1, device=self.device) # standard deviation of depth [m] (or inverse depth?) [1/m], we don't know the scale anyway...


    
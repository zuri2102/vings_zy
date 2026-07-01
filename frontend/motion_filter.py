'''
Main frontend loop start 

rgb frame -> motion filter:
- run extractor, for feature map at low res: fnet
- put into corr, calculate corr between this frame and prev feature maps
- decide whether or not to toss (thresholding, if too small of a difference)--------------------------------------> if drop, next iteration

Want to extract featrues and learn about scene now: cnet
cnet hidden state + feature desciription -> state buffer, now contains the new keyframe's information

----
from VIOFrontend (frontend.py):
self.feature_net = BasicEncoder(output_dim=128, norm_fn='instance') 
self.context_net = BasicEncoder(output_dim=256, norm_fn='none')
self.update_net = UpdateModule()

self.state = StateBuffer(...)
'''
import torch
import lietorch

import ba.math.proj_math as projmath
from nn.corr import AltCorrBlock

class MotionFilter:

    def __init__(self, net, state, device=None):
        # default to cuda if available, otherwise cpu
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.cnet   = net.context_net
        self.fnet   = net.feature_net
        self.update = net.update_net

        self.state  = state
        self.thresh = self.state.motion_filter_thresh
        self.drop_count  = 0  # frames dropped since last keyframe

        self.net = None
        self.inp = None
        self.fmap = None

        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

    def __autocast(self):
        """
        Determines if autocast is used/not used 
        
        cuda/cpu
        """
        if self.device.startswith('cuda'):
            return torch.amp.autocast('cuda', enabled=True)
        import contextlib
        return contextlib.nullcontext() #do nothing


    def __feature_encoder(self, image):
        """
        Run frame through fnet
        
        @return: feature map
        """
        with self.__autocast():
            return self.fnet(image).squeeze(0)


    def __context_encoder(self, image):
        """
        Run frame through cnet

        @return: 1) net (hidden state) and 2) inp 
        """
        with self.__autocast():
            net, inp = self.cnet(image).split([128, 128], dim=2)
            return net.tanh().squeeze(0), inp.relu().squeeze(0)
    
    @torch.no_grad()
    def track(self, tstamp, image, intrinsics=None):
        """
        Part of main loop, updating for new frames
        """
        Id = lietorch.SE3.Identity(1,).data.squeeze() #init pose, identity = default
        H = image.shape[-2] // 8
        W = image.shape[-1] // 8

        # BGR -> RGB, move to gpu, and normalize images to [0,1] range 
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features of this frame
        curr_map = self.__feature_encoder(inputs) 

        # if it is first frame, run cnet to init
        if self.state.kf_idx == 0:
            net, inp = self.__context_encoder(inputs)
            self.net, self.inp, self.fmap = net, inp, curr_map
            self.state.append(tstamp, image, Id, 1.0, self.fmap, self.net, self.inp)

        else:
            grid = projmath.coords_grid(H, W, device=self.device)[None, None]

            corrblock = AltCorrBlock(self.fmap, curr_map) 

            corr = corrblock(grid, 0, 1)

            # run one update step
            _, delta, weight = self.update(self.net, self.inp, corr)

            # check if keyframe meets threshold (determining whether to drop)
            if delta.norm(dim=-1).mean().item() <= self.thresh:
                self.drop_count += 1

            else :
                self.drop_count = 0
                # run through cnet
                net, inp = self.__context_encoder(inputs)
                # update state buffer
                self.net, self.inp, self.fmap = net, inp, curr_map

                self.state.append(tstamp, image, None, None, self.fmap, self.net, self.inp)





            




import torch
import torch.nn as nn
from clipping import GradientClipLayer
from torch_scatter import scatter_mean
from gru import ConvGRU

class FrameSummarizer(nn.Module):
    '''
    egdes (relationships between frames) -> node features (profile for individual frame)
    outputs better parameters
    '''
    def __init__(self, feature_dim=128, up_ratio=8, neigh_dim=3):
        super(FrameSummarizer, self).__init__()
        self.conv1 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1) #raw
        self.conv2 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1) #avged
        self.relu = nn.ReLU(inplace=True)

        self.damp_factor = nn.Sequential(nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1),
                                         GradientClipLayer(), #backwards
                                         nn.Softplus()) #forwards

        
        self.n_chann = up_ratio * up_ratio * neigh_dim * neigh_dim
        self.upmask = nn.Sequential(nn.Conv2d(feature_dim, n_chann, kernel_size=1, padding=0))

        self.feature_dim = feature_dim

    def forward(self, net, i):
        B, N, C, H, W = net.shape
        net = net.view(B*N, C, H, W)

        _, id = torch.unique(i, return_inverse=True)

        net = self.relu(self.conv1(net))

        net = net.view(B, N, self.feature_dim, H, W)
        net = scatter_mean(net, id, dim=1)
        net = net.view(-1, self.feature_dim, H, W)

        net = self.relu(self.conv2(net))

        damp_factor = self.damp_factor(net).view(B, -1, H, W)
        upmask = self.upmask(net).view(B, -1, self.n_chann, H, W)

        return .01 * damp_factor, upmask


class UpdateModule(nn.Module):
    '''
    feature/motion history -> GRU -> prediction refiniment params (step size and mask)

    use_graph_aggregation=True: smoothed, pixels pass information to each other before final prediction
    (aka they cna know if they are part of smae object)
    '''
    def __init__(self, 
                 feature_dim=128, 
                 motion_dim=64, 
                 in_corr_channels=196, 
                 in_motion_channels=4, 
                 output_channels=2,
                 use_graph_aggregation=True):
        super(UpdateModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.output_channels = output_channels
        self.use_graph_aggregation = use_graph_aggregation

        #1) gathering info/features
        
        #summarize visual traj
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(in_corr_channels, feature_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        #summarize velocity traj
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(in_motion_channels, feature_dim, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, motion_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        #translate to physical coordinate corrections (deltax, deltay)
        self.delta = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, output_channels, kernel_size=3, padding=1), #2 ouptus: deltax and deltay
            GradientClipLayer()
        )

        self.weight = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, output_channels, kernel_size=3, padding=1),
            GradientClipLayer(),
            nn.Sigmoid()
        )

        #2) initalize GRU and heads
        n_into_gru = feature_dim + feature_dim + motion_dim
        self.gru = ConvGRU(feature_dim, n_into_gru)

        if self.use_graph_aggregation:
            upsample_channels = (8 * 8) * (3 * 3)  #upsample
            
            #head 1: step size for updates
            self.damp_head = nn.Sequential(
                nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1),
                GradientClipLayer(),
                nn.Softplus()
            )

            #head 2: mask to upscale
            self.upmask_head = nn.Conv2d(feature_dim, upsample_channels, kernel_size=1, padding=0)

    def forward(self, hidden_state, inp, corr, flow=None, edge_indices=None):
        is_graph_input = (len(hidden_state.shape) == 5) #B, N, C, H, W
        
        if is_graph_input: #have multiple batches
            B, N, C, H, W = hidden_state.shape
            output_dim = (B, N, -1, H, W)
            
            hidden_state = hidden_state.view(B*N, -1, H, W)
            inp = inp.view(B*N, -1, H, W)        
            corr = corr.view(B*N, -1, H, W)
            if flow is not None:
                flow = flow.view(B*N, -1, H, W)
        else:
            B, C, H, W = hidden_state.shape
            output_dim = (B, -1, H, W)

        if flow is None: #initalize as 0
            in_motion_ch = self.flow_encoder[0].in_channels
            flow = torch.zeros(hidden_state.shape[0], in_motion_ch, H, W, device=hidden_state.device)

        corr_feats = self.corr_encoder(corr)
        flow_feats = self.flow_encoder(flow)
        hidden_state = self.gru(hidden_state, inp, corr_feats, flow_feats)

        delta = self.delta(hidden_state).view(*output_dim)
        weight = self.weight(hidden_state).view(*output_dim)

        if is_graph_input: #had mutliple batches
            delta = delta.permute(0, 1, 3, 4, 2).contiguous()
            weight = weight.permute(0, 1, 3, 4, 2).contiguous() #B, E, H, W, C
        else:
            delta = delta.permute(0, 2, 3, 1).contiguous()
            weight = weight.permute(0, 2, 3, 1).contiguous()

        if self.use_graph_aggregation and edge_indices is not None and is_graph_input:
            _, relative_indices = torch.unique(edge_indices, return_inverse=True)
            
            #scatter on 4d before making 4d
            node_features_flat = hidden_state.view(B, N, self.feature_dim, H, W)
            node_features_flat = scatter_mean(node_features_flat, relative_indices, dim=1)
            node_features_flat = node_features_flat.view(-1, self.feature_dim, H, W)


            eta = (.01 * self.damp_head(node_features_flat)).view(B, -1, H, W)
            upmask = self.upmask_head(node_features_flat).view(B, -1, self.upmask_head.out_channels, H, W)
            
            return hidden_state, delta, weight, eta, upmask
        
        if is_graph_input:
            hidden_state = hidden_state.view(*output_dim)

        return hidden_state, delta, weight
import torch
import torch.nn as nn


class ConvGRU(nn.Module):
    '''
    h = hidden state/memory map
    i = input

    prev hidden state + new frame -> GRU -> updated hidden state
    '''
    def __init__(self, h_planes=128, i_planes=128):
        super(ConvGRU, self).__init__()
        self.do_checkpoint = False
        self.convz = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convr = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convq = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)

        self.w = nn.Conv2d(h_planes, h_planes, 1, padding=0)

        self.convz_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)
        self.convr_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)
        self.convq_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)

    def forward(self, hidden_state, *inputs):
        inp = torch.cat(inputs, dim=1)
        # net.shape = torch.Size([39, 128, 43, 77])
        hidden_in = torch.cat([net, inp], dim=1)

        b, c, h, w = net.shape
        # net.shape = (48, 128, 43, 77)
        glo = torch.sigmoid(self.w(net)) * net
        glo = glo.view(b, c, h*w).mean(-1).view(b, c, 1, 1)

        z = torch.sigmoid(self.convz(hidden_in) + self.convz_glo(glo))
        r = torch.sigmoid(self.convr(hidden_in) + self.convr_glo(glo))
        q = torch.tanh(self.convq(torch.cat([r*net, inp], dim=1)) + self.convq_glo(glo))

        net = (1-z) * net + z * q
        return net


'''
fnet and cnet

fnet: multidim = False
cnet: mulidim = True
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    '''
    passing data through layers while also putting raw copy, so to not lose original info
    x+y, y = processed, x = raw
    '''
    def __init__(self, in_chann, out_chann, norm_fn='group', stride=1):
        #stride = how many pixels at a time
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_chann, out_chann, kernel_size=3, padding=1, stride=stride) #downsize
        self.conv2 = nn.Conv2d(out_chann, out_chann, kernel_size=3, padding=1) #feature detection/special ops(?)
        self.relu = nn.ReLU(inplace=True)

        num_groups = out_chann // 8 #groups of 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_chann)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_chann)
            if not stride == 1:
                #to normalizing downsamplign raw
                self.norm3 = nn.GroupNorm(numgroups=num_groups, num_channels=out_chann)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(out_chann)
            self.norm2 = nn.BatchNorm2d(out_chann)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(out_chann)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_chann)
            self.norm2 = nn.InstanceNorm2d(out_chann)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(out_chann)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()


        if stride == 1:
            self.downsample = None
        else:
            #run through 1x1 conv and normalize
            self.downsample = nn.Sequential(nn.Conv2d(in_chann, out_chann, kernel_size=1, stride=stride), self.norm3)
    
    def forward(self, x):
        #apply x+y with conv and norm
        y = x
        y = self.conv1(self.relu(self.norm1(y)))
        y = self.conv2(self.relu(self.norm2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return x+y
    
        #CHANGE: norm+relu first, then convolute --> negatives will not explode but are now present

            
class BottleneckBlock(nn.Module):
    '''
    compress, conv, then expand
    simplify computation, good for large amount, but also information can be lost
    '''
    def __init__(self, in_chann, out_chann, norm_fn='group', stride=1):
        #stride = how many pixels at a time
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_chann, out_chann//4, kernel_size=1, padding=0) #compress channels
        self.conv2 = nn.Conv2d(out_chann//4, out_chann//4, kernel_size=3, padding=1, stride=stride) #ops
        self.conv2 = nn.Conv2d(out_chann//4, in_chann, kernel_size=1, padding=0) #expand channels
        self.relu = nn.ReLU(inplace=True)

        num_groups = out_chann // 8 #groups of 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_chann//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_chann//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_chann)
            if not stride == 1:
                #to normalizing downsamplign raw
                self.norm4 = nn.GroupNorm(numgroups=num_groups, num_channels=out_chann)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(out_chann//4)
            self.norm2 = nn.BatchNorm2d(out_chann//4)
            self.norm3 = nn.BatchNorm2d(out_chann)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(out_chann)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_chann//4)
            self.norm2 = nn.InstanceNorm2d(out_chann//4)
            self.norm3 = nn.InstanceNorm2d(out_chann)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(out_chann)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()


        if stride == 1:
            self.downsample = None
        else:
            #run through 1x1 conv and normalize
            self.downsample = nn.Sequential(nn.Conv2d(in_chann, out_chann, kernel_size=1, stride=stride), self.norm4)
    
    def forward(self, x):
        #apply x+y with conv and norm
        y = x
        y = self.conv1(self.relu(self.norm1(y)))
        y = self.conv2(self.relu(self.norm2(y)))
        y = self.conv3(self.relu(self.norm3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return x+y


class BasicEncoder(nn.Module):
    '''
    implementing blocks into layers, performing feature extraction 
    '''
    def __init__(self, out_chann=128, DIM=32, norm_fn='batch', dropout=0.0, multidim=False, block='residual'):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim

        if block == 'residual':
            self.block_type = ResidualBlock
        elif block == 'bottleneck':
            self.block_type = BottleneckBlock
        

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3) #is 7 good?
        self.relu1 = nn.ReLU(inplace=True)

        self.in_chann = DIM
        self.layer1 = self.make_layer(DIM, stride=1)
        self.layer2 = self.make_layer(2*DIM, stride=2) #cut area in half, doubles channels
        self.layer3 = self.make_layer(4*DIM, stride=2) #again^

        self.conv2 = nn.Conv2d(4*DIM, out_chann, kernel_size=1)

        #downsampling, then upsampling, to give context (like blob) to noiser lines
        if self.multidim:
            #downscale, more channels
            self.layer4 = self.make_layer(8*DIM, stride=2)
            self.layer5 = self.make_layer(16*DIM, stride=2)

            self.in_chann = 8*DIM
            self.layer6 = self.make_layer(8*DIM, stride=1)

            self.in_chann = 4*DIM
            self.layer7 = self.make_layer(4*DIM, stride=1)

            #reduce channel size (scaling up)
            self.red_chann1 = nn.Conv2d(16*DIM, 8*DIM, 1)
            self.red_chann2 = nn.Conv2d(8*DIM, 4*DIM, 1)

            self.conv3 = nn.Conv2d(4*DIM, out_chann, kernel_size=1)


        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        
        #NN setup and layer weights initialilzation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def make_layer(self, out_chann, stride=1):
        layer1 = self.block_type(self.in_chann, out_chann, self.norm_fn, stride=stride)
        layer2 = self.block_type(out_chann, out_chann, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_chann = out_chann
        return nn.Sequential(*layers)
    
    def forward(self, x):
        B, N, C1, H1, W1 = x.shape
        x = x.view(B*N, C1, H1, W1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        if self.multidim:
            x4 = self.layer4(x3)
            x5 = self.layer5(x4)

            #up
            upx5 = self.red_chann1(x5)
            upx5 = F.interpolate(upx5, scale_factor=2, mode='bilinear', align_corners=True) # stretch to match post-layer 4
            fusedx4 = upx5 + x4 #blend sharp and blob
            x6 = self.layer6(fusedx4) #go over again

            #up again
            upx6 = self.red_chann2(x6)
            upx6 = F.interpolate(upx6, scale_factor=2, mode='bilinear', align_corners=True) # stretch to match post-layer 4
            fusedx3 = upx6 + x3 #blend sharp and blob
            x7 = self.layer7(fusedx3) #go over again

            x_f = self.conv3(x7)

        else:
            x_f = self.conv2(x3)
        
        if self.dropout is not None:
            x_f = self.dropout(x_f)
        
        _, C2, H2, W2 = x_f.shape
        return x_f.view(B, N, C2, H2, W2)


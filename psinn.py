#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2022-06-20 22:48:43
LastEditTime: 2022-07-17 00:13:36
Description: file content
'''

from cmath import e
import os
from this import s
import torch
import torch.nn as nn
import torch.optim as optim
# from base_net import *
from torchvision.transforms import *
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        out_channels = 4
        self.args = args
 
        num_inv = 1
        
        operations = []

        current_channel = 4

        for j in range(3):
            b = InvBlockExp(current_channel, current_channel//2)
            operations.append(b)

        for i in range(2):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(3):
                b = InvBlockExp(current_channel, current_channel//2)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, l_ms, b_ms, x_pan, h_ms, rev=False, cal_jacobian=False):
        
        pan_vgg = []
        pan_vgg.append(x_pan)
        pan_vgg.append(F.interpolate(x_pan, scale_factor=1/2, mode='bicubic'))
        pan_vgg.append(F.interpolate(x_pan, scale_factor=1/4, mode='bicubic'))

        if not rev:
            out = h_ms
            for index, op in enumerate(self.operations):
                out = op.forward(out, pan_vgg, rev)

        else:
            out = l_ms
            for index, op in enumerate(reversed(self.operations)):
                out = op.forward(out, pan_vgg, rev)
                
        return out

class InvBlockExp(nn.Module):
    def __init__(self, channel_num=3, channel_split_num=3, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp
        
        if channel_num == 4:
            self.split_len3 = self.split_len2 + 1
        elif channel_num == 16:
            self.split_len3 = self.split_len2 + 1
        elif channel_num == 64:
            self.split_len3 = self.split_len2 + 1
        self.Fs1 = FPNBlock(self.split_len3, self.split_len1)
        self.Ft1 = FPNBlock(self.split_len3, self.split_len1)
        self.Fs2 = FPNBlock(self.split_len3, self.split_len2)
        self.Ft2 = FPNBlock(self.split_len3, self.split_len2)

    def forward(self, x, c1, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if x.shape[1] == 4:
            c = c1[0]
        elif x.shape[1] == 16:
            c = c1[1]
        elif x.shape[1] == 64:
            c = c1[2]

        if not rev: 
            self.s1 = self.clamp * (torch.sigmoid(self.Fs1(torch.cat((x2, c), 1))) * 2 - 1)
            v1 = x1.mul(torch.exp(self.s1)) + self.Ft1(torch.cat((x2, c), 1))
            tmp = self.Fs2(torch.cat((v1, c), 1))
            self.s2 = self.clamp * (torch.sigmoid(tmp) * 2 - 1)
            v2 = x2.mul(torch.exp(self.s2)) + self.Ft2(torch.cat((v1, c), 1))
        else:
            self.s2 = self.clamp * (torch.sigmoid(self.Fs2(torch.cat((x1, c), 1))) * 2 - 1)
            v2 = (x2 - self.Ft2(torch.cat((x1, c), 1))).div(torch.exp(self.s2))
            self.s1 = self.clamp * (torch.sigmoid(self.Fs1(torch.cat((v2, c), 1))) * 2 - 1)
            v1 = (x1 - self.Ft1(torch.cat((v2, c), 1))).div(torch.exp(self.s1))

        return torch.cat((v1, v2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

######################################
#               fpn model
######################################

class FPNBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(FPNBlock, self).__init__()
        gc = 32
        bias = True
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        #initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5
        
import torch.nn.init as init
def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, c, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, c, rev=False):
        return self.last_jac

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet
            
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]

        self.vgg256 = nn.Sequential(*modules[:3])
        self.vgg128 = nn.Sequential(*modules[:8])
        self.vgg64 = nn.Sequential(*modules[:18])
        
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        catx = torch.cat([x,x,x],1)
        x256 = self.vgg256(catx)
        x128 = self.vgg128(catx)
        x64 = self.vgg64(catx)
        return [x256, x128, x64]

if __name__ == '__main__':       
    x = torch.randn(1,4,64,64)
    y = torch.randn(1,4,256,256)
    z = torch.randn(1,1,256,256)
    arg = []
    Net = Net(arg)
    out = Net(x, y, z, y)
    print(out.shape)

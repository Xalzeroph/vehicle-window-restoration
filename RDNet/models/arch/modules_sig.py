

# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class LayerNorm2d(nn.Module):


    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.weight.numel(),),
                         self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)

class SimpleGate(nn.Module):


    def forward(self, x):
        a, b = x.chunk(2, dim=1)


        aa, bb = a.float(), b.float()
        magnitude = torch.sqrt(aa.square() + bb.square() + 1.e-6)
        return ((2. ** .5) * aa * bb / magnitude).to(dtype=x.dtype)

class NAFBlock(nn.Module):
    def __init__(self, dim, expand_dim, out_dim, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.):
        super().__init__()
        drop_out_rate = 0. 
        dw_channel = expand_dim
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )


        self.sg = SimpleGate()

        ffn_channel = expand_dim
        self.conv4 = nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=out_dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.ones((1, dim, 1, 1)) * layer_scale_init_value, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones((1, dim, 1, 1)) * layer_scale_init_value, requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)


        x = x * (2. * torch.sigmoid(self.sca(x)))
        x = self.conv3(x)

        x = self.dropout1(x)


        y = inp + x * torch.tanh(self.beta)

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * torch.tanh(self.gamma)


def energy_bounded_innovation(reference, proposal):


    d, u = reference.float(), proposal.float()
    denom = 2. * d.square().sum(dim=1, keepdim=True) + u.square().sum(dim=1, keepdim=True)
    numerator = (d * u).sum(dim=1, keepdim=True)
    coef = numerator / denom.clamp_min(1.e-8)
    return reference + (u - coef * d).to(reference.dtype)


class UpSampleConvnext(nn.Module):
    def __init__(self, ratio, inchannel, outchannel):
        super().__init__()
        self.ratio = ratio
        self.channel_reschedule = nn.Sequential(  

                                        nn.Linear(inchannel, outchannel),
                                        LayerNorm(outchannel, eps=1e-6, data_format="channels_last"))
        self.upsample = nn.Upsample(scale_factor=2**ratio, mode='bilinear',
                                    align_corners=False)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.channel_reschedule(x)
        x = x.permute(0, 3, 1, 2)
        
        return self.upsample(x)

class LayerNorm(nn.Module):


    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first", elementwise_affine = True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.elementwise_affine:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):


    def __init__(self, in_channel, hidden_dim, out_channel, kernel_size=3, layer_scale_init_value=1e-6, drop_path= 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=in_channel)
        self.norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channel, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_channel)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channel)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x

class Decoder(nn.Module):
    def __init__(self, depth=[2,2,2,2], dim=[112, 72, 40, 24], block_type = None, kernel_size = 3) -> None:
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.block_type = block_type
        self._build_decode_layer(dim, depth, kernel_size)


        self.ratio_head = nn.Sequential(
            nn.Conv2d(2 * dim[-1], 2 ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(2))

    def _build_decode_layer(self, dim, depth, kernel_size):
        normal_layers = nn.ModuleList()
        upsample_layers = nn.ModuleList()
        proj_layers = nn.ModuleList()

        norm_layer = LayerNorm

        for i in range(1, len(dim)):
            module = [self.block_type(dim[i], dim[i], dim[i], kernel_size) for _ in range(depth[i])]
            normal_layers.append(nn.Sequential(*module))
            upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            proj_layers.append(nn.Sequential(
                nn.Conv2d(dim[i-1], dim[i], 1, 1), 
                norm_layer(dim[i]),

                nn.GELU()
                ))
        for i in range(1, len(dim)):
            module = [self.block_type(dim[i], dim[i], dim[i], kernel_size) for _ in range(depth[i])]
            normal_layers.append(nn.Sequential(*module))
            upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            proj_layers.append(nn.Sequential(
                               nn.Conv2d(dim[i-1], dim[i], 1, 1),
                               norm_layer(dim[i]),
            ))
        self.normal_layers = normal_layers
        self.upsample_layers = upsample_layers
        self.proj_layers = proj_layers

    def _forward_stage(self, stage, x):
        x = self.proj_layers[stage](x)
        x = self.upsample_layers[stage](x)
        return self.normal_layers[stage](x)

    @staticmethod
    def _orthogonal_layers(transmission, reflection):


        common = (transmission + reflection) * (2. ** -.5)
        difference = (transmission - reflection) * (2. ** -.5)
        return common, difference

    def forward(self, c3, c2, c1, c0):


        t = r = c3
        for scale, skip in enumerate((c2, c1, c0)):
            up_t = self._forward_stage(scale, t)
            up_r = self._forward_stage(scale + 3, r)


            common, difference = self._orthogonal_layers(up_t, up_r)
            t, r = common + skip, difference

        t, r = self._orthogonal_layers(t, r)
        return self.ratio_head(torch.cat((t, r), dim=1))

class SimDecoder(nn.Module):
    def __init__(self, in_channel, encoder_stride) -> None:
        super().__init__()
        self.projback = nn.Sequential(
            LayerNorm(in_channel),
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )

    def forward(self, c3):
        return self.projback(c3)
    

class StarReLU(nn.Module):


    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

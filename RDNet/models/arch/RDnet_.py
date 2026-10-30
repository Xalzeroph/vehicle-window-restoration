from pathlib import Path
from models.arch.focalnet import build_focalnet
from models.arch.semantic_occlusion import SemanticOcclusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from models.arch.modules_sig import ConvNextBlock, Decoder, LayerNorm, NAFBlock, UpSampleConvnext, energy_bounded_innovation
from timm.models.layers import trunc_normal_

def _antialias_before_decimation(x):


    h = x.new_tensor((1., 2., 1.))
    weight = ((h[:, None] * h[None, :]) / 16.).view(1, 1, 3, 3)
    weight = weight.expand(x.shape[1], 1, 3, 3)
    return F.conv2d(F.pad(x, (1,1,1,1), mode='replicate'),
                    weight, groups=x.shape[1])


class Fusion(nn.Module):
    def __init__(self, level, channels, first_col) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),
        ) if level in [1, 2, 3] else nn.Identity()
        if not first_col:
            self.up = UpSampleConvnext(1, channels[level + 1], channels[level]) if level in [0, 1, 2] else nn.Identity()
    
    def forward(self, *args):

        c_down, c_up = args


        down = self.down(_antialias_before_decimation(c_down)
                         if self.level else c_down)
        if self.first_col or self.level == 3:
            return down


        return energy_bounded_innovation(down, self.up(c_up))

class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, block_type=ConvNextBlock) -> None:
        super().__init__()
        countlayer = sum(layers[:level])
        expansion = 4
        self.fusion = Fusion(level, channels, first_col)
        modules = [block_type(channels[level], expansion * channels[level], channels[level], kernel_size=kernel_size,
                                 layer_scale_init_value=1e-6, drop_path=dp_rate[countlayer + i]) for i in
                   range(layers[level])]
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x = self.fusion(*args)
        x_clean = self.blocks(x)
        return x_clean


class SubNet(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory, block_type=ConvNextBlock) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level(0, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

        self.level1 = Level(1, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

        self.level2 = Level(2, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

        self.level3 = Level(3, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

    def _forward_nonreverse(self, *args, coefficients):
        x, c0, c1, c2, c3 = args
        a0, a1, a2, a3 = coefficients
        c0 = a0 * c0 + self.level0(x, c1)
        c1 = a1 * c1 + self.level1(c0, c2)
        c2 = a2 * c2 + self.level2(c1, c3)
        c3 = a3 * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def forward(self, *args):


        effective = tuple(torch.sqrt(a.square() + .05 ** 2)
                          for a in (self.alpha0, self.alpha1,
                                    self.alpha2, self.alpha3))
        if self.save_memory and torch.is_grad_enabled():
            def run(*values):
                return self._forward_nonreverse(*values[:5],
                                                coefficients=values[5:])
            return checkpoint(run, *args, *effective,
                              use_reentrant=False, preserve_rng_state=True)
        return self._forward_nonreverse(*args, coefficients=effective)

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

class FullNet_NLP(nn.Module):
    def __init__(self, channels=[32, 64, 96, 128], layers=[2, 3, 6, 3], num_subnet=5,loss_col=4, kernel_size=3, num_classes=1000,
                 drop_path=0.0, save_memory=True, inter_supv=True, head_init_scale=None, pretrained_cols=16) -> None:
        super().__init__()
        self.num_subnet = num_subnet


        channels = list(channels)
        self.channels = tuple(channels)
        self.layers = tuple(layers)
        self.prompt=nn.Sequential(nn.Linear(in_features=6,out_features=512),
                                  StarReLU(),
                                  nn.Linear(in_features=512,out_features=channels[0]),
                                  StarReLU(),
                                  )
        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(layers))]
        for i in range(num_subnet):
            first_col = True if i == 0 else False
            self.add_module(f'subnet{str(i)}', SubNet(
                channels, layers, kernel_size, first_col, 
                dp_rates=dp_rate, save_memory=save_memory,
                block_type=NAFBlock))


        self.decoder = Decoder(depth=[1, 1, 1, 1], dim=list(reversed(channels)),
                               block_type=NAFBlock, kernel_size=3)

        self.apply(self._init_weights)
        self.baseball = build_focalnet('focalnet_L_384_22k_fl4')
        self.baseball_adapter = nn.ModuleList()


        self.baseball_adapter.append(nn.Conv2d(192, channels[0], kernel_size=1))
        for level, width in enumerate(channels):
            self.baseball_adapter.append(nn.Conv2d(192 * (2 ** level),
                                                    width, kernel_size=1))
        self.semantic_occlusion = None
        self.baseball.load_state_dict(torch.load(
            Path(__file__).resolve().parents[2] / 'pretrained/focal.pth',
            map_location='cpu', weights_only=True))
    def enable_semantic_occlusion(self, width=32):
        if self.semantic_occlusion is not None:
            raise RuntimeError("Semantic occlusion head already exists")
        self.semantic_occlusion = SemanticOcclusion(self.channels,width)
        return self

    def train(self, mode=True):


        super().train(mode)
        self.baseball.eval()
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):


        if any(k.startswith('decoder_blocks.') for k in state_dict):
            live = 'decoder_blocks.2.'
            state_dict = {
                ('decoder.' + k[len(live):] if k.startswith(live) else k): v
                for k, v in state_dict.items()
                if k.startswith(live) or
                not k.startswith(('decoder_blocks.', 'stem_comp.'))
            }
        if 'decoder.projback_.0.weight' in state_dict:
            weights = dict(state_dict)
            wt = weights.pop('decoder.projback_.0.weight')
            wr = weights.pop('decoder.projback_2.0.weight')
            bt = weights.pop('decoder.projback_.0.bias')
            br = weights.pop('decoder.projback_2.0.bias')


            weights['decoder.ratio_head.0.weight'] = torch.cat((.5*wt, -.5*wr), dim=1)
            weights['decoder.ratio_head.0.bias'] = .5*(bt-br)
            state_dict = weights
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def project_layers(observed, contrast):


        if contrast.ndim != 4 or contrast.shape[1] != 3:
            raise ValueError('RDNet display estimator needs 3 RGB corrections')
        if observed.shape != contrast.shape:
            raise ValueError('RDNet observed RGB and correction shape mismatch')


        eps = .5 / 255.
        prior = observed.clamp(eps, 1.-eps)
        displayed_transmission = torch.sigmoid(torch.logit(prior) + 4.*contrast)
        signed_nuisance = observed - displayed_transmission
        return torch.cat((signed_nuisance, displayed_transmission), dim=1)

    def forward(self, x_in, alpha, prompt=True, return_semantic=False,
                semantic_gate=True):


        x_cls_out = []
        c0 = c1 = c2 = c3 = 0
        x_base, x_stem = self.baseball(x_in)
        c0, c1, c2, c3 = x_base
        x_stem = self.baseball_adapter[0](x_stem)
        c0, c1, c2, c3 = (
            self.baseball_adapter[1](c0),
            self.baseball_adapter[2](c1),
            self.baseball_adapter[3](c2),
            self.baseball_adapter[4](c3))
        if prompt:


            scores = torch.tanh(self.prompt(alpha))
            gains = torch.exp(scores)
            x = gains.unsqueeze(-1).unsqueeze(-1) * x_stem
        else:
            x = x_stem


        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'subnet{i}')(x, c0, c1, c2, c3)
        contrast = self.decoder(c3, c2, c1, c0)
        semantics = None
        blending = None
        if self.semantic_occlusion is not None:
            semantics, scale, shift, blending = self.semantic_occlusion(
                (c0,c1,c2,c3), contrast.shape[-2:])
            if semantic_gate:
                contrast = self.semantic_occlusion.restore_correction(
                    contrast,scale,shift)
        if return_semantic and semantics is None:
            raise RuntimeError("Semantic occlusion not enabled")
        output = self.project_layers(x_in, contrast)
        if semantic_gate and blending is not None:
            # The mature synthetic-data restoration can be worse than the input
            # on real pairs. Soft blending makes identity a stable default.
            transmission=x_in+blending*(output[:,3:]-x_in)
            output=torch.cat((x_in-transmission,transmission),dim=1)
        if return_semantic:
            return x_cls_out,[output],semantics
        return x_cls_out,[output]

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)

   

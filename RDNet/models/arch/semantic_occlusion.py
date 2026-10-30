"""Dense semantic nuisance classifier with a neutral, learnable recovery modulation.

Labels: 0=valid background, 1=human-shaped nuisance, 2=vegetation-shaped nuisance,
3=other occluding nuisance. 255 is unknown and is excluded from supervision.
The head predicts nuisance source, not all dark pixels or foreground objects.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticOcclusion(nn.Module):
    def __init__(self, channels, width=32, n_classes=4):
        super().__init__()
        self.proj=nn.ModuleList(nn.Sequential(
            nn.Conv2d(c,width,1),nn.GroupNorm(8,width),nn.GELU()
        ) for c in channels)
        self.refine=nn.Sequential(
            nn.Conv2d(width,width,3,padding=1,groups=width),
            nn.Conv2d(width,width,1),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1,groups=width),
            nn.Conv2d(width,width,1),nn.GELU()
        )
        self.classifier=nn.Conv2d(width,n_classes,1)
        # Exactly neutral at construction, so adding this head to trained RDNet
        # does not alter reconstruction until the modulation layer learns.
        self.modulation=nn.Conv2d(n_classes,6,1)
        self.blend=nn.Conv2d(n_classes,1,1)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        nn.init.zeros_(self.blend.weight)
        nn.init.constant_(self.blend.bias,-2.0)

    def forward(self, features, shape):
        spatial=features[0].shape[-2:]
        feat=0
        for x,proj in zip(features,self.proj):
            value=proj(x)
            feat=feat+F.interpolate(value,size=spatial,mode="bilinear",align_corners=False)
        feat=self.refine(feat)
        logits=F.interpolate(self.classifier(feat),size=shape,mode="bilinear",align_corners=False)
        probability=logits.softmax(1)
        scale_shift=self.modulation(probability)
        scale,shift=scale_shift.chunk(2,dim=1)
        blending=torch.sigmoid(self.blend(probability))
        return logits,scale,shift,blending

    @staticmethod
    def restore_correction(correction,scale,shift):
        # capping modulation limits the effect of immature pseudo-labels
        return correction*(1.+.20*torch.tanh(scale))+.20*torch.tanh(shift)

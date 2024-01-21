import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d
from einops import rearrange
from utils import *
from transformer.attention import BidirectionTransformer

class ResNetBlock_3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, BN=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if BN is True:
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.bn2 = nn.BatchNorm3d(out_channels)
        else:
            self.bn1 = nn.Sequential()
            self.bn2 = nn.Sequential()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )
            if BN is True:
                self.bn_down = nn.BatchNorm3d(out_channels)
            else:
                self.bn_down = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
        
class Feature_Aligner(nn.Module):
    def __init__(self, in_channel=256, mid_channel=256, out_channel=32, n_heads=4, depth=4):
        super().__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel

        self.feature_embedding = nn.Sequential(
            nn.Conv2d(self.in_channel, self.mid_channel, kernel_size=1, bias=False),
            ResNetBlock_2D(self.mid_channel, self.mid_channel, stride=1, BN=False)
        )

        self.att = BidirectionTransformer(self.mid_channel, n_heads=n_heads, d_head=self.mid_channel // n_heads, depth=depth, dropout=0.,
            context_dim=self.mid_channel, normalize=True)

        self.feature_embedding_3d = ResNetBlock_3D(self.mid_channel // 8, 16, stride=1, BN=False)

        self.feature_embedding_2d = nn.Sequential(
            nn.Conv2d(3*8*16, self.out_channel, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1),
        )

    def posemb_sincos_2d(self, patches, channel=128, temperature=10000, dtype=torch.float32):
        _, _, h, w, device, dtype = *patches.shape, patches.device, patches.dtype

        y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
        assert (channel % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
        omega = torch.arange(channel // 4, device = device) / (channel // 4 - 1)
        omega = 1. / (temperature ** omega)

        y = y[None] * omega[:, None, None]
        x = x[None] * omega[:, None, None]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=0)

        return pe.type(dtype)

    def forward_2d3d(self, img_feat_src, img_feat_tgt, random_mask=True, mask_ratio=0.25):
        bs, _, h, w = img_feat_src.shape

        img_feat_src = self.feature_embedding(img_feat_src)
        img_feat_tgt = self.feature_embedding(img_feat_tgt)

        pos_encoding_src = self.posemb_sincos_2d(img_feat_src, channel=self.mid_channel)
        pos_encoding_tgt = self.posemb_sincos_2d(img_feat_tgt, channel=self.mid_channel)

        img_feat_src, img_feat_tgt = self.att(img_feat_src + pos_encoding_src[None], img_feat_tgt + pos_encoding_tgt[None])

        img_feat_src = img_feat_src.reshape(bs, self.mid_channel//8, 8, 8, 8)
        img_feat_tgt = img_feat_tgt.reshape(bs, self.mid_channel//8, 8, 8, 8)

        img_feat_src = self.feature_embedding_3d(img_feat_src)
        img_feat_tgt = self.feature_embedding_3d(img_feat_tgt)

        if random_mask is True:
            mask_src = random_masking(img_feat_src, mask_ratio)
            mask_tgt = random_masking(img_feat_tgt, mask_ratio)
            mask_src, mask_tgt = mask_src.reshape(-1, 1, 8, 8, 8), mask_tgt.reshape(-1, 1, 8, 8, 8)
            img_feat_src = img_feat_src * mask_src
            img_feat_tgt = img_feat_tgt * mask_tgt

        return img_feat_src, img_feat_tgt

    def forward_3d2d(self, img_feat):
        bs = img_feat.shape[0]

        img_feat_z = rearrange(img_feat, 'b c d h w -> b (c d) h w')
        img_feat_y = rearrange(img_feat, 'b c d h w -> b (c h) d w')
        img_feat_x = rearrange(img_feat, 'b c d h w -> b (c w) d h')
        img_feat = torch.cat([img_feat_x, img_feat_y, img_feat_z], dim=1)

        img_feat = self.feature_embedding_2d(img_feat)

        img_feat = F.normalize(img_feat, p=2, dim=1).flatten(2)

        return img_feat

class ResNetBlock_2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, BN=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if BN is True:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Sequential()
            self.bn2 = nn.Sequential()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )
            if BN is True:
                self.bn_down = nn.BatchNorm2d(out_channels)
            else:
                self.bn_down = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

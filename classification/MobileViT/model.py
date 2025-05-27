from typing import Union

import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Reduce

def pair(x):
    """Convert a single integer or a tuple into a tuple of two integers."""
    return x if isinstance(x, tuple) else (x, x)

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU()
    )

class MV2Block(nn.Module):
    """MV2 block described in MobileNetV2.
    Paper: https://arxiv.org/pdf/1801.04381
    Based on: https://github.com/tonylins/pytorch-mobilenet-v2
    """

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2], "Stride must be 1 or 2"

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer block described in ViT.
      Paper: https://arxiv.org/abs/2010.11929
      Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MobileViTBlock(nn.Module):
    """MobileViT block.
    Paper: https://arxiv.org/pdf/2104.02176
    """

    def __init__(self, dim, depth, channels, kernel_size, patch_size, mlp_dim, droppout=0.):
        super().__init__()
        self.patch_height, self.patch_width = pair(patch_size)

        self.conv1 = conv_nxn_bn(channels, channels, kernel_size,stride=1)
        self.conv2 = conv_1x1_bn(channels, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, droppout)

        self.conv3 = conv_1x1_bn(dim, channels)
        self.conv4 = conv_nxn_bn(2 * channels, channels, kernel_size, stride=1)

    def forward(self, x):
        y = x.clone()

        # Local representation
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representation
        b, c, h, w = x.shape
        x = rearrange(x, 'b c (h ph) (w pw) -> b (ph pw) (h w) c', ph=self.patch_height, pw=self.patch_width)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) c -> b c (h ph) (w pw)', h=h//self.patch_height, w=w//self.patch_width,
                      ph=self.patch_height, pw=self.patch_width)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), dim=1)
        x = self.conv4(x)
        return x

class MobileViT(nn.Module):
    def __init__(self,
                 image_size: Union[int, tuple] = (256, 256),
                 dims: list = [96, 120, 144],
                 channels: list = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
                 num_classes: int = 1000,
                 expansion: int = 4,
                 kernel_size: int = 3,
                 patch_size: Union[int, tuple] = (2, 2),
                 depths: list = (2, 4, 3)
                 ):
        super().__init__()
        assert len(dims) == 3, "dims must be a tuple of 3"
        assert len(depths) == 3, "depths must be a tuple of 3"

        image_size = pair(image_size)
        patch_size = pair(patch_size)

        image_height, image_width = image_size
        patch_height, patch_width = patch_size
        assert image_height % patch_height == 0, "image height must be divisible by patch height"
        assert image_width % patch_width == 0, "image width must be divisible by patch width"

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(3, init_dim, stride=2)

        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[3], channels[3], 1, expansion))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 2, expansion),
            MobileViTBlock(dims[0], depths[0], channels[5], kernel_size, patch_size, int(dims[0] * 2))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[5], channels[6], 2, expansion),
            MobileViTBlock(dims[1], depths[1], channels[7], kernel_size, patch_size, int(dims[1] * 4))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[7], channels[8], 2, expansion),
            MobileViTBlock(dims[2], depths[2], channels[9], kernel_size, patch_size, int(dims[2] * 4))
        ]))

        self.to_logits = nn.Sequential(
            conv_1x1_bn(channels[-2], last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(channels[-1], num_classes, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        return self.to_logits(x)

def MobileVit_XXS(**kwargs):
    return  MobileViT(
        dims=[64, 80, 96],
        channels=[16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
        **kwargs
    )

def MobileVit_XS(**kwargs):
    return MobileViT(
        dims=[96, 120, 144],
        channels=[16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        **kwargs
    )

def MobileVit_S(**kwargs):
    return MobileViT(
        dims=[144, 192, 240],
        channels=[16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        **kwargs
    )

def MobileVit(model:str='MobileVit_XXS', **kwargs):
    if model == 'MobileVit_XXS':
        return MobileVit_XXS(**kwargs)
    elif model == 'MobileVit_XS':
        return MobileVit_XS(**kwargs)
    elif model == 'MobileVit_S':
        return MobileVit_S(**kwargs)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example usage
    model = MobileVit('MobileVit_XS').to(device)
    # params number
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    img = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 256x256 image
    preds = model(img.to(device))
    print(preds.shape)  # Should output (1, 1000) for the default num_classes

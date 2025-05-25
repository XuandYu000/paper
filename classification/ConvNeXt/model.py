from typing import Callable, Any

import torch
import torch.nn as nn
from timm.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
        (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
        (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
        We use (2) as we find it slightly faster in PyTorch

        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic depth rate. Default: 0.0
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """
    def __init__(self, dim: int, drop_path: float = 0., layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        # self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.pwconv1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        # self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.pwconv2 = nn.Linear(dim * 4, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = rearrange(x, 'b h w c -> b c h w')

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000,
                 depths: list = [3, 3, 9, 3], dims: list = [96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()

        # stem and 3 intermediate downsample layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first')
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # final layer norm and head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)


    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # global average pooling, (N, C, H, W) -> (N, C)
        return self.norm(x.mean(dim=[-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
        The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
        shape (batch_size, height, width, channels) while channels_first corresponds to inputs
        with shape (batch_size, channels, height, width).
        """
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise ValueError("data_format should be either 'channels_last' or 'channels_first'")
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(dim=1, keepdim=True)
            s = (x - u).pow(2).mean(dim=1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x

def ConvNeXt_T(**kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def ConvNeXt_S(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def ConvNeXt_B(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def ConvNeXt_L(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def ConvNeXt_XL(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    return model

if __name__ == "__main__":
    model = ConvNeXt_S()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)  # torch.Size([1, 1000])

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=False)

    # # 参数量
    # model.eval()
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {num_params / 1e6:.2f}M")
    # # 模型结构
    # # print(model)
"""
Vision Transformer (ViT) implementation in PyTorch.
The paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020) describes the architecture.
Arxiv: https://arxiv.org/abs/2010.11929

The code can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
"""
from typing import Union

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout = dropout),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(self.norm(x)) + x
            x = ff(self.norm(x)) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(
        self,
        image_size: Union[int, tuple] = 224,
        patch_size: Union[int, tuple] = 16,
        num_classes: int = 1000,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim = 3072,
        pool = 'cls',
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool must be either cls (cls token) or mean (mean pooling).'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n , _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

def vit_base_patch16(**kwargs):
    model = ViT(
        image_size = 224 if kwargs.get('image_size') is None else kwargs['image_size'],
        patch_size = 16,
        num_classes = 1000 if kwargs.get('num_classes') is None else kwargs['num_classes'],
        dim = 768,
        depth = 12,
        heads = 12,
        mlp_dim = 3072,
    )
    return model

def vit_large_patch16(**kwargs):
    model = ViT(
        image_size = 224 if kwargs.get('image_size') is None else kwargs['image_size'],
        patch_size = 16,
        num_classes = 1000 if kwargs.get('num_classes') is None else kwargs['num_classes'],
        dim = 1024,
        depth = 24,
        heads = 16,
        mlp_dim = 4096,
    )
    return model

def vit_huge_patch14(**kwargs):
    model = ViT(
        image_size = 224 if kwargs.get('image_size') is None else kwargs['image_size'],
        patch_size = 14,
        num_classes = 1000 if kwargs.get('num_classes') is None else kwargs['num_classes'],
        dim = 1280,
        depth = 32,
        heads = 16,
        mlp_dim = 5120,
    )
    return model

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Example usage
    model = vit_base_patch16().to(device)
    # params number
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    img = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    preds = model(img.to(device))
    print(preds.shape)  # Should output (1, 1000) for the default num_classes
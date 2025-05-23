from typing import Callable, Any

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # x = x.view(batch_size, groups, channels_per_group, height, width)
    x = rearrange(x, "b (g c) h w -> b g c h w", g=groups)
    # transpose
    # x = x.transpose(1, 2).contiguous()
    x = rearrange(x, "b g c h w -> b c g h w")
    # flatten
    # x = x.view(batch_size, -1, height, width)
    x = rearrange(x,"b c g h w -> b (c g) h w", g=groups)

    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 2):
            raise ValueError(f"expected stride to be 1 or 2, but got {stride}")
        self.stride = stride

        branch_features = oup // 2

        # 步距为1时，输入通道数必须是2的倍数
        if (self.stride == 1) and (inp != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {inp} and oup {oup} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
        )

    @staticmethod
    def depthwise_conv(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias=False) -> nn.Module:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: list[int],
                 stages_out_channels: list[int],
                 num_classes=1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual
                 ) -> None:
        super().__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected 3 stages_repeats, but got {len(stages_repeats)}")
        if len(stages_out_channels) != 5:
            raise ValueError("expected 5 stages_out_channels, but got {len(stages_out_channels)}")
        self._stages_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stages_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeat, output_channels in zip(stage_names, stages_repeats, self._stages_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeat - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

def _shufflenetv2(*args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)
    return model

# 星号后的参数必须使用关键字参数
def shufflenet_v2_x0_5(**kwargs:Any) -> ShuffleNetV2:
    return _shufflenetv2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 48, 96, 192, 1024],
        **kwargs
    )

def shufflenet_v2_x1_0(**kwargs:Any) -> ShuffleNetV2:
    return _shufflenetv2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 116, 232, 464, 1024],
        **kwargs
    )

def shufflenet_v2_x1_5(**kwargs:Any) -> ShuffleNetV2:
    return _shufflenetv2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 176, 352, 704, 1024],
        **kwargs
    )

def shufflenet_v2_x2_0(**kwargs:Any) -> ShuffleNetV2:
    return _shufflenetv2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 244, 488, 976, 2048],
        **kwargs
    )

if __name__ == "__main__":
    model = shufflenet_v2_x1_0()
    # x = torch.randn(1, 3, 224, 224)
    # y = model(x)
    # print(y.shape)  # torch.Size([1, 1000])

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=False)

    # 参数量
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")
    # 模型结构
    # print(model)


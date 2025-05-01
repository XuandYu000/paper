from typing import Any, Callable, Optional, Union

import torch
from torch import nn, Tensor

__all__ = ['MobileNetV2']

from torchvision.ops import Conv2dNormActivation


# Adjust the input value v to the nearest number that is divisible by divisor.
# Ensure compatibility with hardware constraints or optimization requirements.
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by divisor
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor

    # Adding divisor / 2 to the value is rounded to the nearest multiple of divisor when integer division(//) is applied.
    # For example, if v = 37 and divisor = 8, then:
    #   v + divisor / 2 = 37 + 4 = 41
    #   rounds = int(41) // 8 * 8 = 5 * 8 = 40
    rounds = int(v + divisor / 2) // divisor * divisor
    new_v = max(min_value, rounds)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class InvertedResidual(nn.Module):
    def __init__(
            self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}, expected 1 or 2")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, kernel_size=1))
            layers.append(norm_layer(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[list[list[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """MobileNet V2 main class

        Args:
            num_classes (int): Number of classes for classification. Default is 1000.
            width_mult (float): Width multiplier for the model. Default is 1.0.
            inverted_residual_setting (list[list[int]]): Configuration for inverted residual blocks.
            round_nearest (int): Round nearest value for channel dimensions. Default is 8.
            block (Callable[..., nn.Module]): Block type to use. Default is InvertedResidual.
            norm_layer (Callable[..., nn.Module]): Normalization layer to use. Default is BatchNorm2d.
            dropout (float): Dropout rate. Default is 0.2.
        """
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # t: expand ratio
                # c: output channels
                # n: number of blocks
                # s: stride
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user konws t, c, n, s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}")

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: list[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6),
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    # 为什么定义_forward_impl?
    # The reason for using a separate _forward_impl method instead of directly implementing the logic in the forward method is related to TorchScript limitations.
    # TorchScript, a feature of PyTorch used for model serialization and optimization, does not fully support inheritance in certain cases.
    def _forward_impl(self, x: Tensor) -> Tensor:
        # Forward pass through the network
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        # Override forward method
        return self._forward_impl(x)

if __name__ == "__main__":
    # Example usage
    model = MobileNetV2(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)  # Should be [1, 1000]
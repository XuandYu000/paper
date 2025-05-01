from functools import partial
from typing import Any, Callable, Optional, Union

import torch
from collections.abc import Sequence
from sympy.geometry.entity import scale
from torch import nn, Tensor
from torchvision.ops import Conv2dNormActivation, SqueezeExcitation as SElayer

__all__ = ["MobileNetV3", "mobilenet_v3_small", "mobilenet_v3_large"]

# Adjust the input value v to the nearest number that is divisible by divisor.
# Ensure compatibility with hardware constraints or optimization requirements.
def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
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

class InvertedResidualConig:
    # Stores information listed at Table1 and 2 of MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float
    ):
        self.input_channels = _make_divisible(input_channels * width_mult, 8)
        self.kernel = kernel
        self.expanded_channels = _make_divisible(expanded_channels * width_mult, 8)
        self.out_channels = _make_divisible(out_channels * width_mult, 8)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

class InvertedResidual(nn.Module):
    def __init__(
            self,
            cnf: InvertedResidualConig,
            norm_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value, must be 1 or 2")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: list[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.input_channels != cnf.expanded_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.input_channels * 4, 8)
            layers.append(
                se_layer(cnf.expanded_channels, squeeze_channels)
            )

        # project
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result

class MobileNetV3(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: list[InvertedResidualConig],
            last_channel: int,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.2,
            **kwargs: Any,
    ) -> None:
        """
        MobileNetV3
        Args:
            inverted_residual_setting: List of inverted residual configurations.
            last_channel: Number of channels in the last layer.
            num_classes: Number of classes for classification.
            block: Block type to use for the model.
            norm_layer: Normalization layer to use.
            dropout: Dropout probability.
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("inverted_residual_setting is empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all(isinstance(x, InvertedResidualConig) for x in inverted_residual_setting)
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

        layers: list[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(
                block(cnf, norm_layer=norm_layer)
            )

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _mobilenet_v3_conf(
    arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConig, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = _make_divisible(1280 // reduce_divider * width_mult)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = _make_divisible(1024 // reduce_divider * width_mult)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel

def _mobilenet_v3(
    inverted_residual_setting: list[InvertedResidualConig],
    last_channel: int,
    **kwargs: Any
) -> MobileNetV3:
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    return model

def mobilenet_v3_small(
    **kwargs: Any
) -> MobileNetV3:
    """
    Constructs a MobileNetV3-Small model.
    Args:
        width_mult: Width multiplier for the model.
        reduced_tail: Whether to reduce the tail of the model.
        dilated: Whether to use dilated convolutions.
        **kwargs: Additional keyword arguments.
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_small"
    )
    return _mobilenet_v3(inverted_residual_setting, last_channel, **kwargs)

def mobilenet_v3_large(
    **kwargs: Any
) -> MobileNetV3:
    """
    Constructs a MobileNetV3-Large model.
    Args:
        width_mult: Width multiplier for the model.
        reduced_tail: Whether to reduce the tail of the model.
        dilated: Whether to use dilated convolutions.
        **kwargs: Additional keyword arguments.
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_large"
    )
    return _mobilenet_v3(inverted_residual_setting, last_channel, **kwargs)
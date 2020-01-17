# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
All custom models used
"""
from typing import List, Dict, Any

from torch import nn

_in_channels_key = "in_channels"
_out_channels_key = "out_channels"
_kernel_size_key = "kernel_size"
_stride_key = "stride"
_dilation_key = "dilation"
_padding_key = "padding"


class BaseDilatedNet(nn.Module):
    _layers_params = None

    def __init__(self, in_channels: int = 1, num_classes: int = 1,
                 layers_params: List[Dict[str, Any]] = None):
        super().__init__()
        if layers_params is None:
            layers_params = self._layers_params

        assert all(_out_channels_key in l for l in layers_params), f"'out_channels' must be specified"
        assert all(_kernel_size_key in l for l in layers_params), f"'kernel_size' must be specified"
        layers_in_channels = [in_channels] + [l_params.get("out_channels") for l_params in layers_params[:-1]]
        self.net = nn.Sequential(
            *[self._get_layer(in_channels=ch_in, **layer_params)
              for ch_in, layer_params in zip(layers_in_channels, layers_params)],
            # logits prediction
            nn.Conv2d(
                in_channels=layers_params[-1][_out_channels_key],
                out_channels=num_classes,
                kernel_size=1)
        )

    def _get_layer(self, in_channels, out_channels, kernel_size, activation="relu", use_bn=False, **kwargs):
        if activation.lower() != "relu":
            raise NotImplementedError("other activations not supported")
        if _padding_key not in kwargs:
            if kernel_size > 1:
                if kernel_size % 2 == 0:
                    raise NotImplementedError("implemented correctly only for odd kernel sizes for now")
                dilation = 1
                if _dilation_key in kwargs:
                    dilation = kwargs[_dilation_key]
                if isinstance(dilation, int):
                    dilation = (dilation, dilation)
                mul = (kernel_size - 1) // 2
                padding = tuple([mul * d for d in dilation])
                kwargs[_padding_key] = padding
        sublayers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, **kwargs),
            nn.ReLU(inplace=True)
        ]
        if use_bn:
            sublayers.append(nn.BatchNorm2d(out_channels, momentum=0.1))
        return nn.Sequential(*sublayers)

    def forward(self, x):
        return self.net.forward(x)


class ZharkovDilatedNet(BaseDilatedNet):
    _channels = 24

    _layers_params = [
        # downscale module (note that it is NOT separable)
        {_out_channels_key: _channels, _kernel_size_key: 3, _stride_key: 2},
        {_out_channels_key: _channels, _kernel_size_key: 3, _stride_key: 1},
        {_out_channels_key: _channels, _kernel_size_key: 3, _stride_key: 2},
        # context module
        {_out_channels_key: _channels, _kernel_size_key: 3, _dilation_key: 1},
        {_out_channels_key: _channels, _kernel_size_key: 3, _dilation_key: 2},
        {_out_channels_key: _channels, _kernel_size_key: 3, _dilation_key: 4},
        {_out_channels_key: _channels, _kernel_size_key: 3, _dilation_key: 8},
        {_out_channels_key: _channels, _kernel_size_key: 3, _dilation_key: 16},
        {_out_channels_key: _channels, _kernel_size_key: 3, _dilation_key: 1},
    ]

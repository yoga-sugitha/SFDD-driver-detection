import torch.nn as nn
import torch

act_fn_by_name = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class Conv2DBlock(nn.Module):
    """
    Double Conv2D block with BatchNorm. Downsamples via strided first conv.

    Structure:
        Conv(stride=2) -> BN -> Act -> Conv(stride=1) -> BN -> Act

    The stride=2 in the first conv replaces MaxPool for downsampling.
    This is learnable (unlike MaxPool) and is the standard approach in
    modern residual networks.

    Args:
        c_in: Number of input channels
        c_out: Number of output channels
        act_fn: Activation function class
        kernel_size: Convolution kernel size (default 3)
        stride: Stride of first conv for spatial downsampling (default 2)
    """
    def __init__(self, c_in: int, c_out: int, act_fn, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
        )

    def forward(self, x):
        return self.net(x)


class ResidualUnit(nn.Module):
    """
    Standard pre-activation style residual unit with optional skip projection.

    Structure:
        Conv -> BN -> Act -> Conv -> BN
        + skip (projected if subsample=True)
        -> Act

    When subsample=True, uses stride=2 in the first conv and a 1x1 projection
    on the skip connection to match spatial size and channel count.

    Args:
        c_in: Number of input channels
        c_out: Number of output channels
        act_fn: Activation function class
        subsample: If True, stride=2 + project skip (use when c_in != c_out)
    """
    def __init__(self, c_in: int, c_out: int, act_fn, subsample: bool = False):
        super().__init__()
        stride = 2 if subsample else 1
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
            if subsample else None
        )
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        skip = self.downsample(x) if self.downsample is not None else x
        return self.act_fn(z + skip)


class InterleavedGroup(nn.Module):
    """
    One interleaved group: [Conv2DBlock x conv_per_group] + [ResidualUnit].

    This is the core repeating unit of the architecture. Each group:
      1. Runs N conv blocks to learn and compress features
      2. Runs one residual unit to stabilize gradients via skip connection

    The residual unit does NOT subsample here (same channels in/out) — its
    job is gradient protection, not downsampling. Downsampling is handled
    by the strided Conv2DBlocks.

    Args:
        c_in: Input channels to this group
        c_out: Output channels (applied to all conv blocks and the residual)
        act_fn: Activation function class
        conv_per_group: How many Conv2DBlocks before the ResidualUnit
    """
    def __init__(self, c_in: int, c_out: int, act_fn, conv_per_group: int = 2):
        super().__init__()
        layers = []
        for i in range(conv_per_group):
            layers.append(Conv2DBlock(
                c_in if i == 0 else c_out,
                c_out,
                act_fn=act_fn,
                stride=2 if i == 0 else 1,  # only first conv downsamples
            ))
        # Residual at same channel width — no subsampling
        layers.append(ResidualUnit(c_out, c_out, act_fn=act_fn, subsample=False))
        self.group = nn.Sequential(*layers)

    def forward(self, x):
        return self.group(x)


class ConvResNet(nn.Module):
    """
    Fully interleaved Conv + Residual network.

    Architecture overview:
    ┌─────────────────────────────────────────────────────┐
    │ Stem: Conv7x7(stride=2) -> BN -> Act -> MaxPool     │
    ├─────────────────────────────────────────────────────┤
    │ Group 1: Conv(64->128) -> Conv(128) -> Res(128)     │
    │ Group 2: Conv(128->256) -> Conv(256) -> Res(256)    │
    │ Group 3: Conv(256->512) -> Conv(512) -> Res(512)    │
    ├─────────────────────────────────────────────────────┤
    │ Head: AdaptiveAvgPool -> Flatten -> Linear          │
    └─────────────────────────────────────────────────────┘

    Total conv layers: 2 per group x 3 groups = 6 ✓
    Total residual units: 1 per group x 3 groups = 3

    Key design decisions vs previous version:
      - No more "all conv then all residual" — fully interleaved
      - Channel progression starts at 64 (stem output), never shrinks
      - No mid_pool — AdaptiveAvgPool2d handles any spatial size safely
      - conv_per_group and group_channels are both configurable

    Args:
        num_classes: Number of output classes
        group_channels: Output channels for each group (len = num groups)
        conv_per_group: Conv2DBlocks per group before the residual unit
        act_fn_name: One of 'relu', 'gelu', 'tanh'
        input_channels: Input image channels (3 for RGB)
    """
    def __init__(
        self,
        num_classes: int = 10,
        group_channels: list = None,
        conv_per_group: int = 2,
        act_fn_name: str = "relu",
        input_channels: int = 3,
        **kwargs,
    ):
        super().__init__()

        if group_channels is None:
            group_channels = [128, 256, 512]  # 3 groups -> 6 conv layers total

        self.num_classes = num_classes
        self.group_channels = group_channels
        self.conv_per_group = conv_per_group
        self.act_fn_name = act_fn_name
        self.act_fn = act_fn_by_name[act_fn_name]
        self.input_channels = input_channels

        self._create_network()
        self._init_params()

    def _create_network(self):
        # Stem — same as before, outputs (B, 64, H/4, W/4)
        self.input_net = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            self.act_fn(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Interleaved groups
        groups = []
        c_in = 64
        for c_out in self.group_channels:
            groups.append(InterleavedGroup(
                c_in, c_out,
                act_fn=self.act_fn,
                conv_per_group=self.conv_per_group,
            ))
            c_in = c_out
        self.groups = nn.Sequential(*groups)

        # Head — AdaptiveAvgPool safely handles any spatial size
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_in, self.num_classes),
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity=self.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)   # (B, 64,  H/4,  W/4)
        x = self.groups(x)      # (B, 512, H/32, W/32) approx
        x = self.output_net(x)  # (B, num_classes)
        return x

# class Conv2DBlock(nn.Module):
#     def __init__(self, in_channels, filters, kernel_size=3, strides=1, padding=1, activation='relu') -> None:
#         super().__init__()
#         self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()

#         self.layers = nn.Sequential(
#             nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=strides, padding=padding),
#             nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=strides, padding=padding),
#             nn.BatchNorm2d(filters),
#             nn.MaxPool2d(2,2)
#         )
#     def forward(self, x):
#         return self.activation(self.layers(x))
    
# class ResidualUnit(nn.Module):
#     def __init__(self, in_channels, filters, kernel_size=3, strides=1, padding=1, activation='relu'):
#         super().__init__()
#         self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()
        
#         self.main_layers = nn.Sequential(
#             nn.Conv2d(in_channels, filters, kernel_size, stride=strides, padding=padding),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Conv2d(filters, filters, kernel_size, stride=strides, padding=padding),
#             nn.BatchNorm2d(filters)
#         )

#         self.skip_layers = nn.Identity()
#         if strides > 1 or in_channels != filters:
#             self.skip_layers = nn.Sequential(
#                 nn.Conv2d(in_channels, filters, kernel_size=1, stride=strides, padding=0),
#                 nn.BatchNorm2d(filters)
#             )

#     def forward(self, x):
#         main_out = self.main_layers(x)
#         skip_out = self.skip_layers(x)
#         return self.activation(main_out+skip_out)

# class ResidualModel(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
        
#         # Initial layers
#         self.rescale = lambda x: x / 255.0 # Can be done during the dataset creation
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # Conv2DBlock layers
#         self.conv_blocks = nn.ModuleList()
#         in_channels = 64
#         for filters in [16, 32, 64, 128, 256]:
#             self.conv_blocks.append(Conv2DBlock(in_channels, filters))
#             in_channels = filters
        
#         # ResidualUnit layers
#         self.res_units = nn.ModuleList()
#         prev_filters = 256
#         for filters in [256] * 6 + [512] * 3:
#             strides = 1 if filters == prev_filters else 2
#             self.res_units.append(ResidualUnit(prev_filters, filters, strides=strides))
#             prev_filters = filters
        
#         # Final layers
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(512, num_classes)
#         self.softmax = nn.Softmax(dim=1)
    
#     def forward(self, x):
#         # Rescaling
#         x = self.rescale(x)
        
#         # Initial conv block
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
        
#         # Conv2D blocks
#         for block in self.conv_blocks:
#             x = block(x)
        
#         # Residual units
#         for unit in self.res_units:
#             x = unit(x)
        
#         # Global average pooling and classification
#         x = self.global_avg_pool(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         x = self.softmax(x)
        
#         return x

# act_fn_by_name = {
#     "relu": nn.ReLU,
#     "gelu": nn.GELU,
#     "tanh": nn.Tanh,
# }


# class Conv2DBlock(nn.Module):
#     """
#     Double Conv2D block with BatchNorm and MaxPool.

#     Args:
#         c_in: Number of input channels
#         c_out: Number of output channels
#         act_fn: Activation function class
#         kernel_size: Convolution kernel size
#         stride: Convolution stride
#     """
#     def __init__(self, c_in: int, c_out: int, act_fn, kernel_size: int = 3, stride: int = 1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
#             nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
#             nn.BatchNorm2d(c_out),
#             act_fn(),
#             nn.MaxPool2d(kernel_size=2),
#         )

#     def forward(self, x):
#         return self.net(x)


# class ResidualUnit(nn.Module):
#     """
#     Residual unit with optional skip connection projection.

#     Args:
#         c_in: Number of input channels
#         c_out: Number of output channels
#         act_fn: Activation function class
#         subsample: Whether to downsample (stride=2) and project skip connection
#     """
#     def __init__(self, c_in: int, c_out: int, act_fn, subsample: bool = False):
#         super().__init__()
#         stride = 2 if subsample else 1
#         self.net = nn.Sequential(
#             nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(c_out),
#             act_fn(),
#             nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(c_out),
#         )
#         self.downsample = (
#             nn.Sequential(
#                 nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(c_out),
#             )
#             if subsample else None
#         )
#         self.act_fn = act_fn()

#     def forward(self, x):
#         z = self.net(x)
#         skip = self.downsample(x) if self.downsample is not None else x
#         return self.act_fn(z + skip)


# class ConvResNet(nn.Module):
#     """
#     Hybrid Conv2DBlock + ResidualUnit network from scratch.

#     Args:
#         num_classes: Number of output classes
#         conv_channels: List of channel dims for Conv2DBlock stages
#         res_channels: List of channel dims for ResidualUnit stages
#         act_fn_name: Name of activation function
#         input_channels: Number of input image channels
#     """
#     def __init__(
#         self,
#         num_classes: int = 10,
#         conv_channels: list = None,
#         res_channels: list = None,
#         act_fn_name: str = "relu",
#         input_channels: int = 3,
#         **kwargs,
#     ):
#         super().__init__()

#         if conv_channels is None:
#             conv_channels = [16, 32, 64, 128, 256]
#         if res_channels is None:
#             res_channels = [256] * 6 + [512] * 3

#         self.num_classes = num_classes
#         self.conv_channels = conv_channels
#         self.res_channels = res_channels
#         self.act_fn_name = act_fn_name
#         self.act_fn = act_fn_by_name[act_fn_name]
#         self.input_channels = input_channels

#         self._create_network()
#         self._init_params()

#     def _create_network(self):
#         # Stem
#         self.input_net = nn.Sequential(
#             nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             self.act_fn(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         )

#         # Conv2DBlock stages
#         conv_blocks = []
#         c_in = 64
#         for c_out in self.conv_channels:
#             conv_blocks.append(Conv2DBlock(c_in, c_out, act_fn=self.act_fn))
#             c_in = c_out
#         self.conv_blocks = nn.Sequential(*conv_blocks)

#         # ResidualUnit stages
#         res_blocks = []
#         prev_c = c_in
#         for c_out in self.res_channels:
#             subsample = c_out != prev_c
#             res_blocks.append(ResidualUnit(prev_c, c_out, act_fn=self.act_fn, subsample=subsample))
#             prev_c = c_out
#         self.res_blocks = nn.Sequential(*res_blocks)

#         # Output
#         self.output_net = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(prev_c, self.num_classes),
#         )

#     def _init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=self.act_fn_name)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.input_net(x)
#         x = self.conv_blocks(x)
#         x = self.res_blocks(x)
#         x = self.output_net(x)
#         return x

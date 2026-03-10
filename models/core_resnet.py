#### ==== fully interleaved version

"""
core_resnet_v3.py

Architecture: FullyInterleavedConvResNet
-----------------------------------------
Evolution from v2 (InterleavedConvResNet):
  v1: [All Conv] -> [All Residual]          # bad: over-compresses before residuals
  v2: [Conv, Conv, Res] x 3 groups         # better: residuals every 2 convs per group
  v3: Conv->Conv->Res->Conv->Conv->Res->    # best: single flat interleaved sequence
      Conv->Conv->Res  (no group abstraction)

The key difference between v2 and v3:
  v2 resets channel width at the start of each group, creating implicit "stages"
  v3 treats the whole network as one continuous interleaved sequence, giving
  the optimizer more freedom to find useful cross-block representations.

Additional improvements over v2:
  1. Squeeze-and-Excitation (SE) in each ResidualUnit
     - Learns per-channel attention weights (which channels matter more)
     - Adds ~2% params but consistently improves accuracy
     - Used in EfficientNet, SENet, ResNeXt variants

  2. Zero-init last BN gamma in residual branch
     - Each residual block starts as identity at init: F(x) + x ≈ x
     - Stabilizes early training, allows higher LR
     - From "Bag of Tricks for Image Classification" (He et al.)

  3. Dropout before classifier head
     - Prevents over-reliance on specific feature channels
     - Cheap regularization, often +0.5-1.5% accuracy

Full architecture (default):
┌──────────────────────────────────────────────────────────────┐
│ Stem: Conv7x7(s2) -> BN -> ReLU -> MaxPool(s2)              │
│       output: (B, 64, H/4, W/4)                             │
├──────────────────────────────────────────────────────────────┤
│ Conv2DBlock(64  -> 128, stride=2)                           │
│ Conv2DBlock(128 -> 128, stride=1)                           │
│ ResidualUnit (128 -> 128, SE) ← skip connection checkpoint  │
├──────────────────────────────────────────────────────────────┤
│ Conv2DBlock(128 -> 256, stride=2)                           │
│ Conv2DBlock(256 -> 256, stride=1)                           │
│ ResidualUnit (256 -> 256, SE) ← skip connection checkpoint  │
├──────────────────────────────────────────────────────────────┤
│ Conv2DBlock(256 -> 512, stride=2)                           │
│ Conv2DBlock(512 -> 512, stride=1)                           │
│ ResidualUnit (512 -> 512, SE) ← skip connection checkpoint  │
├──────────────────────────────────────────────────────────────┤
│ Head: AdaptiveAvgPool(1,1) -> Dropout -> Flatten -> Linear  │
└──────────────────────────────────────────────────────────────┘
Total conv layers : 6  (2 per segment)
Total residual units: 3  (1 per segment, after every 2 convs)
"""

import torch
import torch.nn as nn

act_fn_by_name = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).

    Learns a per-channel scaling vector from global context:
      1. Squeeze: global average pool -> (B, C, 1, 1) -> (B, C)
      2. Excitation: FC -> Act -> FC -> Sigmoid -> (B, C) scale weights
      3. Scale: multiply input channels by learned weights

    This costs very few parameters (2 small FC layers) but lets the network
    dynamically emphasize useful channels and suppress noisy ones.

    Args:
        channels: Number of input/output channels
        reduction: Bottleneck ratio for the excitation FC layers (default 16)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)  # floor at 4 to avoid degenerate case
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, C, H, W) -> (B, C, 1, 1)
            nn.Flatten(),                      # (B, C)
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale                       # channel-wise rescaling


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Conv2DBlock(nn.Module):
    """
    Double Conv2D block with BatchNorm and activation.

    Structure:
        Conv(stride) -> BN -> Act -> Conv(stride=1) -> BN -> Act

    The first conv handles spatial downsampling (stride=2) or preserves
    spatial size (stride=1). Second conv always stride=1.

    Args:
        c_in: Input channels
        c_out: Output channels
        act_fn: Activation function class
        kernel_size: Conv kernel size (default 3)
        stride: Stride of first conv (default 2 for downsampling)
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
    Residual unit with SE attention and zero-init last BN.

    Structure:
        Conv(3x3) -> BN -> Act -> Conv(3x3) -> BN -> SE -> (+skip) -> Act

    Improvements over basic residual unit:
      - SE block applied to the residual branch output before addition
      - Last BN gamma zero-initialized so block starts as identity
      - Optional skip projection when channels change or stride=2

    Args:
        c_in: Input channels
        c_out: Output channels
        act_fn: Activation function class
        subsample: Stride=2 + project skip if True (use when c_in != c_out)
        use_se: Whether to apply Squeeze-and-Excitation (default True)
        se_reduction: SE bottleneck ratio (default 16)
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        act_fn,
        subsample: bool = False,
        use_se: bool = True,
        se_reduction: int = 16,
    ):
        super().__init__()
        stride = 2 if subsample else 1

        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),   # gamma zero-inited below
        )

        # SE on the residual branch (before addition)
        self.se = SEBlock(c_out, reduction=se_reduction) if use_se else nn.Identity()

        # Skip projection when spatial or channel size changes
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
            if subsample else None
        )

        self.act_fn = act_fn()

        # Zero-init last BN gamma so each residual block starts as identity.
        # F(x) + x ≈ x at init -> more stable gradients early in training.
        nn.init.constant_(self.net[-1].weight, 0)

    def forward(self, x):
        z = self.se(self.net(x))
        skip = self.downsample(x) if self.downsample is not None else x
        return self.act_fn(z + skip)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ConvResNet(nn.Module):
    """
    Fully interleaved Conv + Residual network (v3).

    Places one ResidualUnit after every `res_every` Conv2DBlocks, producing
    a flat alternating sequence rather than grouped stages. This gives skip
    connections the densest possible coverage across the downsampling pipeline.

    With default settings (6 conv blocks, res_every=2):
        Conv -> Conv -> Res -> Conv -> Conv -> Res -> Conv -> Conv -> Res

    Configurable parameters allow easy ablations:
      - Increase res_every to 3 for sparser residuals (2 res blocks total)
      - Increase group_channels depth for a wider model
      - Toggle use_se to ablate the SE contribution
      - Tune dropout_p for more/less regularization

    Args:
        num_classes: Number of output classes
        group_channels: Channel widths per segment, len = number of [Conv,Conv,Res] groups
        res_every: Number of Conv2DBlocks before each ResidualUnit (default 2)
        act_fn_name: Activation function name ('relu', 'gelu', 'tanh')
        input_channels: Input image channels
        use_se: Enable Squeeze-and-Excitation in residual units
        se_reduction: SE bottleneck ratio
        dropout_p: Dropout probability before classifier head (0 to disable)
    """
    def __init__(
        self,
        num_classes: int = 10,
        group_channels: list = None,
        res_every: int = 2,
        act_fn_name: str = "relu",
        input_channels: int = 3,
        use_se: bool = True,
        se_reduction: int = 16,
        dropout_p: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        if group_channels is None:
            group_channels = [128, 256, 512]  # 3 segments -> 6 conv layers

        self.num_classes = num_classes
        self.group_channels = group_channels
        self.res_every = res_every
        self.act_fn_name = act_fn_name
        self.act_fn = act_fn_by_name[act_fn_name]
        self.input_channels = input_channels
        self.use_se = use_se
        self.se_reduction = se_reduction
        self.dropout_p = dropout_p

        self._create_network()
        self._init_params()

    def _create_network(self):
        # Stem
        self.input_net = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            self.act_fn(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Build flat interleaved sequence:
        # for each channel group: [Conv(down), Conv(same)] + [Res(same)]
        layers = []
        c_in = 64
        for c_out in self.group_channels:
            # res_every conv blocks per segment
            for i in range(self.res_every):
                layers.append(Conv2DBlock(
                    c_in if i == 0 else c_out,
                    c_out,
                    act_fn=self.act_fn,
                    stride=2 if i == 0 else 1,   # only first conv in segment downsamples
                ))
            # one residual unit at the same channel width — gradient checkpoint
            layers.append(ResidualUnit(
                c_out, c_out,
                act_fn=self.act_fn,
                subsample=False,        # no extra downsampling — convs already did it
                use_se=self.use_se,
                se_reduction=self.se_reduction,
            ))
            c_in = c_out

        self.backbone = nn.Sequential(*layers)

        # Head
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=self.dropout_p),   # regularize before linear
            nn.Linear(c_in, self.num_classes),
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity=self.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                # Note: last BN in each ResidualUnit.net already zero-inited
                # in ResidualUnit.__init__, so we only set weight=1 here for
                # BNs that haven't been explicitly overridden.
                if m.weight is not None and m.weight.data.sum() == 0:
                    pass  # already zero-inited by ResidualUnit, leave it
                else:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)   # (B, 64,   H/4,   W/4)
        x = self.backbone(x)    # (B, 512,  H/32,  W/32) approx
        x = self.output_net(x)  # (B, num_classes)
        return x

#### === end of codes


# import torch.nn as nn
# import torch


# ## below is a version that has performance slightly better than the original one
# act_fn_by_name = {
#     "relu": nn.ReLU,
#     "gelu": nn.GELU,
#     "tanh": nn.Tanh,
# }


# class Conv2DBlock(nn.Module):
#     """
#     Double Conv2D block with BatchNorm. Downsamples via strided first conv.

#     Structure:
#         Conv(stride=2) -> BN -> Act -> Conv(stride=1) -> BN -> Act

#     The stride=2 in the first conv replaces MaxPool for downsampling.
#     This is learnable (unlike MaxPool) and is the standard approach in
#     modern residual networks.

#     Args:
#         c_in: Number of input channels
#         c_out: Number of output channels
#         act_fn: Activation function class
#         kernel_size: Convolution kernel size (default 3)
#         stride: Stride of first conv for spatial downsampling (default 2)
#     """
#     def __init__(self, c_in: int, c_out: int, act_fn, kernel_size: int = 3, stride: int = 2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride,
#                       padding=kernel_size // 2, bias=False),
#             nn.BatchNorm2d(c_out),
#             act_fn(),
#             nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1,
#                       padding=kernel_size // 2, bias=False),
#             nn.BatchNorm2d(c_out),
#             act_fn(),
#         )

#     def forward(self, x):
#         return self.net(x)


# class ResidualUnit(nn.Module):
#     """
#     Standard pre-activation style residual unit with optional skip projection.

#     Structure:
#         Conv -> BN -> Act -> Conv -> BN
#         + skip (projected if subsample=True)
#         -> Act

#     When subsample=True, uses stride=2 in the first conv and a 1x1 projection
#     on the skip connection to match spatial size and channel count.

#     Args:
#         c_in: Number of input channels
#         c_out: Number of output channels
#         act_fn: Activation function class
#         subsample: If True, stride=2 + project skip (use when c_in != c_out)
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


# class InterleavedGroup(nn.Module):
#     """
#     One interleaved group: [Conv2DBlock x conv_per_group] + [ResidualUnit].

#     This is the core repeating unit of the architecture. Each group:
#       1. Runs N conv blocks to learn and compress features
#       2. Runs one residual unit to stabilize gradients via skip connection

#     The residual unit does NOT subsample here (same channels in/out) — its
#     job is gradient protection, not downsampling. Downsampling is handled
#     by the strided Conv2DBlocks.

#     Args:
#         c_in: Input channels to this group
#         c_out: Output channels (applied to all conv blocks and the residual)
#         act_fn: Activation function class
#         conv_per_group: How many Conv2DBlocks before the ResidualUnit
#     """
#     def __init__(self, c_in: int, c_out: int, act_fn, conv_per_group: int = 2):
#         super().__init__()
#         layers = []
#         for i in range(conv_per_group):
#             layers.append(Conv2DBlock(
#                 c_in if i == 0 else c_out,
#                 c_out,
#                 act_fn=act_fn,
#                 stride=2 if i == 0 else 1,  # only first conv downsamples
#             ))
#         # Residual at same channel width — no subsampling
#         layers.append(ResidualUnit(c_out, c_out, act_fn=act_fn, subsample=False))
#         self.group = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.group(x)


# class ConvResNet(nn.Module):
#     """
#     Fully interleaved Conv + Residual network.

#     Architecture overview:
#     ┌─────────────────────────────────────────────────────┐
#     │ Stem: Conv7x7(stride=2) -> BN -> Act -> MaxPool     │
#     ├─────────────────────────────────────────────────────┤
#     │ Group 1: Conv(64->128) -> Conv(128) -> Res(128)     │
#     │ Group 2: Conv(128->256) -> Conv(256) -> Res(256)    │
#     │ Group 3: Conv(256->512) -> Conv(512) -> Res(512)    │
#     ├─────────────────────────────────────────────────────┤
#     │ Head: AdaptiveAvgPool -> Flatten -> Linear          │
#     └─────────────────────────────────────────────────────┘

#     Total conv layers: 2 per group x 3 groups = 6 ✓
#     Total residual units: 1 per group x 3 groups = 3

#     Key design decisions vs previous version:
#       - No more "all conv then all residual" — fully interleaved
#       - Channel progression starts at 64 (stem output), never shrinks
#       - No mid_pool — AdaptiveAvgPool2d handles any spatial size safely
#       - conv_per_group and group_channels are both configurable

#     Args:
#         num_classes: Number of output classes
#         group_channels: Output channels for each group (len = num groups)
#         conv_per_group: Conv2DBlocks per group before the residual unit
#         act_fn_name: One of 'relu', 'gelu', 'tanh'
#         input_channels: Input image channels (3 for RGB)
#     """
#     def __init__(
#         self,
#         num_classes: int = 10,
#         group_channels: list = None,
#         conv_per_group: int = 2,
#         act_fn_name: str = "relu",
#         input_channels: int = 3,
#         **kwargs,
#     ):
#         super().__init__()

#         if group_channels is None:
#             group_channels = [128, 256, 512]  # 3 groups -> 6 conv layers total

#         self.num_classes = num_classes
#         self.group_channels = group_channels
#         self.conv_per_group = conv_per_group
#         self.act_fn_name = act_fn_name
#         self.act_fn = act_fn_by_name[act_fn_name]
#         self.input_channels = input_channels

#         self._create_network()
#         self._init_params()

#     def _create_network(self):
#         # Stem — same as before, outputs (B, 64, H/4, W/4)
#         self.input_net = nn.Sequential(
#             nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             self.act_fn(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         )

#         # Interleaved groups
#         groups = []
#         c_in = 64
#         for c_out in self.group_channels:
#             groups.append(InterleavedGroup(
#                 c_in, c_out,
#                 act_fn=self.act_fn,
#                 conv_per_group=self.conv_per_group,
#             ))
#             c_in = c_out
#         self.groups = nn.Sequential(*groups)

#         # Head — AdaptiveAvgPool safely handles any spatial size
#         self.output_net = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(c_in, self.num_classes),
#         )

#     def _init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out",
#                                         nonlinearity=self.act_fn_name)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.input_net(x)   # (B, 64,  H/4,  W/4)
#         x = self.groups(x)      # (B, 512, H/32, W/32) approx
#         x = self.output_net(x)  # (B, num_classes)
#         return x

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

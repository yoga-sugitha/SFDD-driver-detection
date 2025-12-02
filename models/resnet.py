"""
ResNet implementation from scratch
"""
import torch.nn as nn

act_fn_by_name = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh
}

class ResNetBlock(nn.Module):
    """
    ResNet residual block
    
    Args:
        c_in: Number of input channels
        act_fn: Activation function class
        subsample: Whether to downsample spatial dimensions
        c_out: Number of output channels (only used if subsample=True)
    """
    def __init__(self, c_in: int, act_fn, subsample: bool = False, c_out: int = -1):
        super().__init__()
        if not subsample:
            c_out = c_in

        self.net = nn.Sequential(
            nn.Conv2d(
                c_in, c_out, 
                kernel_size=3, 
                padding=1, 
                stride=1 if not subsample else 2, 
                bias=False
            ),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

        self.downsample = (
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) 
            if subsample else None
        )
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return self.act_fn(out)


class ResNet(nn.Module):
    """
    ResNet architecture from scratch
    
    Args:
        num_classes: Number of output classes
        num_blocks: List of number of blocks per stage
        c_hidden: List of channel dimensions per stage
        act_fn_name: Name of activation function
    """
    def __init__(
        self,
        num_classes: int = 10,
        num_blocks: list = None,
        c_hidden: list = None,
        act_fn_name: str = "relu",
        **kwargs
    ):
        super().__init__()
        
        if num_blocks is None:
            num_blocks = [3, 3, 3]
        if c_hidden is None:
            c_hidden = [16, 32, 64]
            
        self.num_classes = num_classes
        self.c_hidden = c_hidden
        self.num_blocks = num_blocks
        self.act_fn_name = act_fn_name
        self.act_fn = act_fn_by_name[act_fn_name]
        self.block_class = ResNetBlock
        
        self._create_network()
        self._init_params()

    def _create_network(self):
        """Build the network architecture"""
        c_hidden = self.c_hidden

        # Input layer
        self.input_net = nn.Sequential(
            nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_hidden[0]),
            self.act_fn(),
        )

        # ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                subsample = bc == 0 and block_idx > 0
                blocks.append(
                    self.block_class(
                        c_in=c_hidden[block_idx if not subsample else (block_idx - 1)],
                        act_fn=self.act_fn,
                        subsample=subsample,
                        c_out=c_hidden[block_idx],
                    )
                )
        self.blocks = nn.Sequential(*blocks)

        # Output layer
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Linear(c_hidden[-1], self.num_classes)
        )

    def _init_params(self):
        """Initialize model parameters"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode="fan_out", 
                    nonlinearity=self.act_fn_name
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x

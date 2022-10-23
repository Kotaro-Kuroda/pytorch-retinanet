from typing import Callable, Dict, List, Optional, Union

from torch import nn, Tensor
from torchvision.ops import misc as misc_nn_ops
from retinanet.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from retinanet.bifpn import BiFPN


class BackboneWithBiFPN(nn.Module):
    def __init__(
        self,
        body,
        in_channels_list: List[int],
        out_channels: int,
        num_layers=2,
        epsilon=1e-4,
        extra_block=True
    ) -> None:
        super().__init__()

        """if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()"""

        self.body = body
        self.fpn = BiFPN(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            num_layers=num_layers,
            epsilon=epsilon,
            extra_block=extra_block
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        return x


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        body,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        """if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()"""

        self.body = body
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        return x

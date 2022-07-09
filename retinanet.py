from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2, RetinaNet, RetinaNetHead
from torchvision.models.detection.rpn import AnchorGenerator
import poolformer
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from functools import partial
from torch import nn


def get_model(num_classes):
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                         for x in [16, 32, 64, 128, 256])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    model = retinanet_resnet50_fpn_v2(num_classes=num_classes,
                                      pretrained=False, pretrained_backbone=True, progress=True, topk_candidates=1000, detections_per_img=500)
    model.anchor_generator = anchor_generator
    return model


def poolformer_backbone_model(num_classes):


<< << << < HEAD
== == == =
return_layers = {f'norm{k}': v for v, k in enumerate([2, 4, 6])}
plformer = poolformer.poolformer_s36(fork_feat=True, return_layers=return_layers)

in_channles_list = [128, 320, 512]
>>>>>> > f6aaae0d6abbb870008ee5aa006a42c68486c29c
out_channels = 256
backbone = BackboneWithFPN(plformer, return_layers, in_channles_list, out_channels, extra_blocks=LastLevelP6P7(512, 256))
backbone.body = plformer
anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                     for x in [16, 32, 64, 128, 256])
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
anchor_generator = AnchorGenerator(
    anchor_sizes, aspect_ratios
)
head = RetinaNetHead(
    backbone.out_channels,
    anchor_generator.num_anchors_per_location()[0],
    num_classes,
    norm_layer=partial(nn.GroupNorm, 32),
)
head.regression_head._loss_type = "giou"
model = RetinaNet(backbone, num_classes=num_classes, anchor_generator=anchor_generator, head=head, topk_candidates=1000, detections_per_img=500)
return model

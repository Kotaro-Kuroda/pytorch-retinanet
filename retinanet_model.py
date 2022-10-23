from retinanet.efficientnet import EfficientNet
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2, RetinaNet, RetinaNetHead
from torchvision.models.detection.rpn import AnchorGenerator
from retinanet import poolformer
from retinanet.backbone_utils import BackboneWithFPN, BackboneWithBiFPN
from retinanet.feature_pyramid_network import LastLevelP6P7
from functools import partial
from torch import nn
from torchvision.models.detection import fcos_resnet50_fpn, FCOS


def fcos(num_classes, pretrained_backbone=False):
    return fcos_resnet50_fpn(num_classes=num_classes, pretrained_backbone=pretrained_backbone)


def get_model(num_classes, pretrained=True, layers=[0, 2, 4, 6], arch='retinanet', backbone='resnet50', out_channels=256, baseline='fpn', num_layers=2, **kwargs):
    if arch == 'retinanet':
        if backbone == 'resnet50':
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                                 for x in [16, 32, 64, 128, 256, 512])
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
            model = retinanet_resnet50_fpn_v2(num_classes=num_classes,
                                              pretrained=False, pretrained_backbone=pretrained, progress=True, **kwargs)
            model.anchor_generator = anchor_generator
        elif 'efficientnet' in backbone:
            backbone = efficientnet_backbone(backbone, pretrained, layers, out_channels, baseline, num_layers)
            anchor_size = [2 ** (4 + i) for i in range(len(layers))]
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in anchor_size)
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
            model = RetinaNet(backbone, num_classes=num_classes, anchor_generator=anchor_generator, head=head, **kwargs)
        elif 'poolformer' in backbone:
            anchor_size = [2 ** (4 + i) for i in range(len(layers) + 2)]
            backbone = poolformer_backbone(backbone=backbone, pretrained=pretrained, layers=layers, out_channels=out_channels)
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in anchor_size)
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
            model = RetinaNet(backbone, num_classes=num_classes, anchor_generator=anchor_generator, head=head, **kwargs)
    elif arch == 'fcos':
        if backbone == 'resnet50':
            model = fcos_resnet50_fpn(num_classes=num_classes, pretrained_backbone=pretrained)
        else:
            backbone = poolformer_backbone(backbone=backbone, pretrained=pretrained)
            model = FCOS(backbone, num_classes, **kwargs)
    return model


def poolformer_backbone(backbone, pretrained, layers, out_channels=160):
    return_layers = {f'norm{k}': v for v, k in enumerate(layers)}
    if backbone == 'poolformer_s12':
        plformer = poolformer.poolformer_s12(pretrained=pretrained, fork_feat=True, return_layers=return_layers)
    elif backbone == 'poolformer_s24':
        plformer = poolformer.poolformer_s24(pretrained=pretrained, fork_feat=True, return_layers=return_layers)
    elif backbone == 'poolformer_s36':
        plformer = poolformer.poolformer_s36(pretrained=pretrained, fork_feat=True, return_layers=return_layers)
    in_channles_list = []
    for layer in layers:
        if layer == 0:
            in_channles_list.append(64)
        elif layer == 2:
            in_channles_list.append(128)
        elif layer == 4:
            in_channles_list.append(320)
        elif layer == 6:
            in_channles_list.append(512)
    # backbone = BackboneWithBiFPN(plformer, in_channles_list, out_channels, num_layers=6, extra_block=True)
    # backbone = BackboneWithFPN(plformer, in_channles_list, out_channels, extra_blocks=LastLevelP6P7(max(in_channles_list), out_channels))
    return backbone


def efficientnet_backbone(backbone, pretrained, layers, out_channels, baseline, num_layers):
    if backbone == 'efficientnet_b0':
        in_channels_list = [16, 24, 40, 80, 112, 192, 320]
    elif backbone == 'efficientnet_b1':
        in_channels_list = [16, 24, 40, 80, 112, 192, 320]
    elif backbone == 'efficientnet_b2':
        in_channels_list = [16, 24, 48, 88, 120, 208, 352]
    elif backbone == 'efficientnet_b3':
        in_channels_list = [24, 32, 48, 96, 136, 232, 384]
    elif backbone == 'efficientnet_b4':
        in_channels_list = [24, 32, 56, 112, 160, 272, 448]
    elif backbone == 'efficientnet_b5':
        in_channels_list = [24, 40, 64, 128, 176, 304, 512]
    elif backbone == 'efficientnet_b6':
        in_channels_list = [32, 40, 72, 144, 200, 344, 576]
    elif backbone == 'efficientnet_b7':
        in_channels_list = [32, 48, 80, 160, 224, 384, 640]
    in_channels_list = [in_channels_list[i - 1] for i in layers]
    backbone = EfficientNet(backbone, layers, pretrained=pretrained)
    if baseline == 'bifpn':
        backbone = BackboneWithBiFPN(backbone, in_channels_list, out_channels=out_channels, num_layers=num_layers, extra_block=False)
    else:
        backbone = BackboneWithFPN(backbone, in_channels_list, out_channels, None)
    return backbone

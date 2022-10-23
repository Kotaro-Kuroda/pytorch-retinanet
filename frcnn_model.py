
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNet
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
import torch
import torch.nn.functional as F
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def retinanet_resnet50_fpn(pretrained=False, progress=True,
                           num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    model_urls = {
        'retinanet_resnet50_fpn_coco':
        'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth',
    }
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, returned_layers=[1, 2, 3],
                                   extra_blocks=LastLevelP6P7(256, 256), trainable_layers=trainable_backbone_layers)
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                         for x in [16, 32, 64, 128, 256])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    model = RetinaNet(backbone, num_classes,
                      anchor_generator=anchor_generator, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['retinanet_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model


def frcnn_model(nn_model, num_classes):
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256)), aspect_ratios=((0.5, 1.0, 2.0)))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2)
    if nn_model == 'resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False, box_detections_per_img=1000, pretrained_backbone=True, rpn_batch_size_per_image=1000, box_batch_size_per_image=1000)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)
        """trainable_backbone_layers = _validate_trainable_layers(
            True, None, 5, 3)
        backbone = resnet_fpn_backbone(
            'resnet50', False, trainable_layers=trainable_backbone_layers)
        model = FasterRCNN(backbone, num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator, box_detections_per_img=500, box_roi_pool=roi_pooler)"""

    elif nn_model == 'mobilenet_v3_large':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=False, rpn_anchor_generator=anchor_generator, box_detections_per_img=500, pretrained_backbone=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)
        """backbone = torchvision.models.mobilenet_v3_large(
            pretrained=True).features
        backbone.out_channels = 960
        model = FasterRCNN(backbone, num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator, box_detections_per_img=500, box_roi_pool=roi_pooler)"""

    elif nn_model == 'retinanet':
        model = retinanet_resnet50_fpn_v2(num_classes=num_classes,
                                          pretrained=False, pretrained_backbone=True, progress=True, topk_candidates=1000, detections_per_img=500)

    return model

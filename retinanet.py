import torchvision


def model(dataset_class):
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        pretrained_backbone=True, num_classes=len(dataset_class) + 1)
    return model

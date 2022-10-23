from torch import nn
from timm import models
import torch.nn.functional as F
from collections import OrderedDict
import torch


class EfficientNet(nn.Module):
    def __init__(self, level, fork_feat, pretrained=False):
        super(EfficientNet, self).__init__()
        self.fork_feat = fork_feat
        if level == 'efficientnet_b0':
            backbone = models.efficientnet_b0(pretrained)
        elif level == 'efficientnet_b1':
            backbone = models.efficientnet_b1(pretrained)
        elif level == 'efficientnet_b2':
            backbone = models.efficientnet_b2(pretrained)
        elif level == 'efficientnet_b3':
            backbone = models.efficientnet_b3(pretrained)
        elif level == 'efficientnet_b4':
            backbone = models.efficientnet_b4(pretrained)
        elif level == 'efficientnet_b5':
            backbone = models.efficientnet_b5(pretrained)
        elif level == 'efficientnet_b6':
            backbone = models.efficientnet_b6(pretrained)
        elif level == 'efficientnet_b7':
            backbone = models.efficientnet_b7(pretrained)
        elif level == 'efficientnet_b8':
            backbone = models.efficientnet_b8(pretrained)
        else:
            raise ValueError()
        self.conv_stem = backbone.conv_stem
        self.bn1 = backbone.bn1
        self.blocks = backbone.blocks

        self.layer1 = backbone.blocks[0]
        self.layer2 = backbone.blocks[1]
        self.layer3 = backbone.blocks[2]
        self.layer4 = backbone.blocks[3]
        self.layer5 = backbone.blocks[4]
        self.layer6 = backbone.blocks[5]
        self.layer7 = backbone.blocks[6]

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        out = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) in self.fork_feat:
                out[f'p{(i+1)}'] = x
        return OrderedDict(out)


def main():
    for i in range(8):
        level = f'efficientnet_b{i}'
        backbone = EfficientNet(level, [0, 1, 2, 3, 4, 5, 6, 7])
        a = torch.randn(1, 3, 1024, 1024)
        out = backbone(a)
        print(level)
        for key in out.keys():
            print(out[key].shape)
        print()


if __name__ == "__main__":
    main()

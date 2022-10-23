import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple


class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding, dilation, groups=in_channels, bias=True)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    def __init__(
        self,
        num_feats: int,
        out_channels: int,
        epsilon=1e-4
    ):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for i in range(num_feats):
            if i < num_feats - 1:
                inner_block = DepthwiseConvBlock(out_channels, out_channels)
                self.inner_blocks.append(inner_block)
            if i > 0:
                layer_block = DepthwiseConvBlock(out_channels, out_channels)
                self.layer_blocks.append(layer_block)

        self.weights = nn.ParameterDict()
        self.weights['w1'] = nn.Parameter(torch.ones(2, num_feats - 2))
        self.weights['w2'] = nn.Parameter(torch.ones(3, num_feats - 2))
        self.weights['w0'] = nn.Parameter(torch.ones(2))
        self.weights['w3'] = nn.Parameter(torch.ones(2))
        self.w1_relu = nn.ReLU(inplace=False)
        self.w2_relu = nn.ReLU(inplace=False)
        self.w0_relu = nn.ReLU(inplace=False)
        self.w3_relu = nn.ReLU(inplace=False)

    def forward(self, x):
        w1 = self.w1_relu(self.weights['w1'])
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        w2 = self.w2_relu(self.weights['w2'])
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        w0 = self.w0_relu(self.weights['w0'])
        w0 = w0 / (torch.sum(w0, dim=0) + self.epsilon)
        w3 = self.w3_relu(self.weights['w3'])
        w3 = w3 / (torch.sum(w3, dim=0) + self.epsilon)

        # top down
        results_td = []
        last_inner = x[-1]
        results_td.append(last_inner)
        for i in range(len(x) - 2, -1, -1):
            p = x[i]
            module = self.inner_blocks[i]
            if i > 0:
                p_td = module(w1[0, i - 1] * p + w1[1, i - 1] * F.interpolate(last_inner, size=p.shape[-2:], mode="nearest"))
            else:
                p_td = module(w0[0] * p + w0[1] * F.interpolate(last_inner, size=p.shape[-2:], mode="nearest"))
            last_inner = p_td
            results_td.append(p_td)

        # bottom up
        results_bu = []
        results_bu.append(results_td[-1])
        last_inner = results_td[-1]
        for i in range(len(results_td) - 2, -1, -1):
            p_td = results_td[i]
            module = self.layer_blocks[i]
            if i < len(results_td) - 1:
                p_out = module(w2[0, i - 1] * x[len(x) - 1 - i] + w2[1, i - 1] * p_td + w2[2, i - 1] * F.interpolate(last_inner, p_td.shape[-2:], mode='nearest'))
            else:
                p_out = module(w3[0] * x[len(x) - 1 - i] + w3[1] * F.interpolate(last_inner, x[i].shape[-2:], mode='nearest'))
            last_inner = p_out
            results_bu.append(p_out)
        return results_bu


class BiFPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int, num_layers: int = 2, epsilon: float = 1e-4, extra_block=True):
        super(BiFPN, self).__init__()
        self.input_layer = nn.ModuleList()
        for in_channels in in_channels_list:
            layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            self.input_layer.append(layer)
        self.extra_block = extra_block
        if self.extra_block:
            self.p6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        bifpns = []
        num_feats = len(in_channels_list) + 2 if self.extra_block else len(in_channels_list)
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(num_feats, out_channels, epsilon))
        self.bifpn = nn.Sequential(*bifpns)

    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())
        results = []
        for i, module in enumerate(self.input_layer):
            results.append(module(x[i]))
        last = x[-1]
        if self.extra_block:
            p6 = self.p6(last)
            p7 = self.p7(p6)
            results.append(p6)
            results.append(p7)
            names.append('p6')
            names.append('p7')
        result_bifpn = self.bifpn(results)
        out = OrderedDict([(k, v) for k, v in zip(names, result_bifpn)])
        return out


def main():
    num_feats = 4
    out_channels = 256
    keys = [str(i) for i in range(num_feats)]
    in_channels_list = [2 ** (5 + i) for i in range(num_feats)]
    x = {}
    for key in keys:
        feat = torch.randn(1, 2 ** (5 + int(key)), 2 ** (10 - int(key)), 2 ** (10 - int(key)))
        x[key] = feat

    bf = BiFPN(in_channels_list, out_channels)
    out = bf(x)
    print(out)


if __name__ == "__main__":
    main()

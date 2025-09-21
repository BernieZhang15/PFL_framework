import torch
from torch import Tensor
import torch.nn as nn
from typing import Any, Callable, List, Optional
from .Bayes_layers.Bayes_conv import Bayes_Conv2d
from .Bayes_layers.Bayes_linear import Bayes_Linear


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1, device=None):
    """3x3 convolution with padding"""
    return Bayes_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False,
                        dilation=dilation, device=device)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, device=None):
    """1x1 convolution"""
    return Bayes_Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, device=device)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_bn=True,
            device=None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride, device=device)
        if has_bn:
            self.bn1 = norm_layer(planes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, device=device)
        if has_bn:
            self.bn2 = norm_layer(planes)
        else:
            self.bn2 = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: BasicBlock,
            layers: List[int],
            features: List[int] = [64, 128, 256, 512],
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_bn=True,
            bn_block_num=4,
            device=None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Bayes_Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, device=device)
        if has_bn:
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = []
        self.layers.extend(self._make_layer(block, 64, layers[0], has_bn=has_bn and (bn_block_num > 0), device=device))
        for num in range(1, len(layers)):
            self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2,
                                                dilate=replace_stride_with_dilation[num - 1],
                                                has_bn=has_bn and (num < bn_block_num), device=device))

        for i, layer in enumerate(self.layers):
            setattr(self, f'layer_{i}', layer)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = Bayes_Linear(features[len(layers) - 1] * block.expansion, num_classes, device=device)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: BasicBlock, planes: int, blocks: int, stride: int = 1, dilate: bool = False,
                    has_bn=True, device=None) -> List:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if has_bn:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride, device=device),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride, device=device),
                    nn.Identity(),
                )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer, has_bn, device)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, has_bn=has_bn, device=device))

        return layers

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.layers)):
            layer = getattr(self, f'layer_{i}')
            x = layer(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return self._forward_impl(x), kl


def resnet34(**kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet18(**kwargs: Any) -> ResNet:  # 18 = 2 + 2 * (2 + 2 + 2 + 2)
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet10(**kwargs: Any) -> ResNet:  # 10 = 2 + 2 * (1 + 1 + 1 + 1)
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet8(**kwargs: Any) -> ResNet:  # 8 = 2 + 2 * (1 + 1 + 1)
    return ResNet(BasicBlock, [1, 1, 1], **kwargs)


def resnet6(**kwargs: Any) -> ResNet:  # 6 = 2 + 2 * (1 + 1)
    return ResNet(BasicBlock, [1, 1], **kwargs)


def resnet4(**kwargs: Any) -> ResNet:  # 4 = 2 + 2 * (1)
    return ResNet(BasicBlock, [1], **kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random_sample = torch.randn([32, 3, 32, 32]).to(device)

    model = resnet18(device=device).to(device)

    output, kl = model(random_sample)

    for n, p in model.named_parameters():
        print(n)

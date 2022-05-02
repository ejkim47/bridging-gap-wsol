"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

from .method import AttentiveDrop
from .util import remove_layer
from .util import replace_layer
from .util import initialize_weights

__all__ = ['resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNetCam(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNetCam, self).__init__()

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            return {'logits': logits, 'cams': cams}
        return {'logits': logits}

    def _make_layer(self, block, planes, blocks, stride, use_latter=False):
        if use_latter:
            self._layer_latter(block, planes, blocks)
        else:
            layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def _layer_latter(self, block, planes, blocks):
        self.inplanes = planes * block.expansion
        layers = []
        for _ in range(blocks):
            layers.append(block(self.inplanes, planes))

        return layers


class ResNetDrop(ResNetCam):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNetDrop, self).__init__(block, layers, num_classes=num_classes,
                                         large_feature_map=large_feature_map)
        stride_l3 = 1 if large_feature_map else 2
        del self.layer1, self.layer2, self.layer3, self.layer4
        self.inplanes = 64

        self.drop_layer = AttentiveDrop(kwargs['drop_threshold'], kwargs['drop_prob'])

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4_1 = self._make_layer(block, 512, 1, stride=1)
        self.layer4_2 = self._make_layer(block, 512, layers[3]-1, stride=1)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        unerased_x = self.layer4_1(x)
        erased_x = self.drop_layer(unerased_x)

        unerased_x = self.layer4_2(unerased_x)
        erased_x = self.layer4_2(erased_x)

        pre_logit = self.avgpool(unerased_x)
        pre_logit = pre_logit.view(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        x_normalized = F.normalize(unerased_x, dim=1)
        weight_normalized = F.normalize(self.fc.weight, dim=1)
        sim = F.conv2d(input=x_normalized, weight=weight_normalized.view(*weight_normalized.shape, 1, 1))

        if return_cam:
            feature_map = unerased_x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            return {'logits': logits, 'cams': cams,
                    'feature': unerased_x, 'feature_erased': erased_x, 'sim': sim}
        return {'logits': logits, 'feature': unerased_x, 'feature_erased': erased_x, 'sim': sim}

def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )

def batch_replace_layer_drop(state_dict):
    state_dict = replace_layer(state_dict, 'layer4.0.', 'layer4_1.0.')
    state_dict = replace_layer(state_dict, 'layer4.1.', 'layer4_2.0.')
    state_dict = replace_layer(state_dict, 'layer4.2.', 'layer4_2.1.')
    return state_dict


def load_pretrained_model(model, wsol_method, path=None, **kwargs):
    strict_rule = True

    if path:
        state_dict = torch.load(os.path.join(path, 'resnet50.pth'))
    else:
        state_dict = load_url(model_urls['resnet50'], progress=True)

    if 'drop' in wsol_method and isinstance(model, ResNetDrop):
        state_dict = batch_replace_layer_drop(state_dict)

    if kwargs['dataset_name'] != 'ILSVRC':
        state_dict = remove_layer(state_dict, 'fc')
        strict_rule = False

    model.load_state_dict(state_dict, strict=strict_rule)
    return model


def resnet50(architecture_type, pretrained=False, pretrained_path=None,
             **kwargs):
    model = {'cam': ResNetCam,
             'drop': ResNetDrop}[architecture_type](Bottleneck, [3, 4, 6, 3],
                                                  **kwargs)
    if pretrained:
        model = load_pretrained_model(model, architecture_type,
                                      path=pretrained_path, **kwargs)
    return model

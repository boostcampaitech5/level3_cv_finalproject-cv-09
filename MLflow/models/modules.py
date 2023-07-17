import torch
import torch.nn as nn
import functools


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    
class ModuleHelper(object):

    @staticmethod
    def BNReLU(num_features, bn_type='torchbn', **kwargs):
        if bn_type == 'torchbn':
            return nn.Sequential(
                nn.BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'torchsyncbn':
            return nn.Sequential(
                nn.SyncBatchNorm(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'gn':
            return nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=num_features, **kwargs),
                nn.ReLU()
            )
        else:
            exit(1)

    @staticmethod
    def BatchNorm2d(bn_type='torchbn', ret_cls=False):
        if bn_type == 'torchbn':
            return nn.BatchNorm2d

        elif bn_type == 'torchsyncbn':
            return nn.SyncBatchNorm

        elif bn_type == 'gn':
            return functools.partial(nn.GroupNorm, num_groups=32)    

        else:
            # Log.error('Not support BN type: {}.'.format(bn_type))
            exit(1)

    @staticmethod
    def load_model(model, pretrained=None, all_match=True, network='resnet101'):
        if pretrained is None:
            return model

        if all_match:
            # Log.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained)
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if 'resinit.{}'.format(k) in model_dict:
                    load_dict['resinit.{}'.format(k)] = v
                else:
                    load_dict[k] = v
            model.load_state_dict(load_dict)

        else:
            # Log.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained)

            model_dict = model.state_dict()

            if network == "hrnet_plus":
                # pretrained_dict['conv1_full_res.weight'] = pretrained_dict['conv1.weight']
                # pretrained_dict['conv2_full_res.weight'] = pretrained_dict['conv2.weight']
                load_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict.keys()}

            else:
                load_dict = {'.'.join(k.split('.')[1:]): v \
                             for k, v in pretrained_dict.items() \
                             if '.'.join(k.split('.')[1:]) in model_dict}

            # used to debug
            # if int(os.environ.get("debug_load_model", 0)):
            #     Log.info('Matched Keys List:')
            #     for key in load_dict.keys():
            #         Log.info('{}'.format(key))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out

            
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * 4, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


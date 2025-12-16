# Copyright (c) Tencent Inc. All rights reserved.
import copy
from typing import List, Union
import math

import torch
import torch.nn as nn
from torch import Tensor
from mmdet.utils import ConfigType, OptMultiConfig

from mmdet.registry import MODELS
from .yolov8_pafpn import YOLOv8PAFPN
from torch.nn.parameter import Parameter
from torch.nn.modules.batchnorm import _BatchNorm


activation_table = {'relu':nn.ReLU(),
                    'silu':nn.SiLU(),
                    'hardswish':nn.Hardswish()
                    }

def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x

class SiLU(nn.Module):
    '''Activation of SiLU'''
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class ConvModule_torch(nn.Module):
    '''A combination of Conv + BN + Activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_type, padding=None, groups=1, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation_type is not None:
            self.act = activation_table.get(activation_type)
        self.activation_type = activation_type

    def forward(self, x):
        if self.activation_type is None:
            return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        if self.activation_type is None:
            return self.conv(x)
        return self.act(self.conv(x))


class ConvBNReLU(nn.Module):
    '''Conv and BN with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule_torch(in_channels, out_channels, kernel_size, stride, 'relu', padding, groups, bias)

    def forward(self, x):
        return self.block(x)


class ConvBNSiLU(nn.Module):
    '''Conv and BN with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule_torch(in_channels, out_channels, kernel_size, stride, 'silu', padding, groups, bias)

    def forward(self, x):
        return self.block(x)


class ConvBN(nn.Module):
    '''Conv and BN without activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule_torch(in_channels, out_channels, kernel_size, stride, None, padding, groups, bias)

    def forward(self, x):
        return self.block(x)


class ConvBNHS(nn.Module):
    '''Conv and BN with Hardswish activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule_torch(in_channels, out_channels, kernel_size, stride, 'hardswish', padding, groups, bias)

    def forward(self, x):
        return self.block(x)


class SPPFModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNReLU):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = block(in_channels, c_, 1, 1)
        self.cv2 = block(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class SimSPPF(nn.Module):
    '''Simplified SPPF with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNReLU):
        super().__init__()
        self.sppf = SPPFModule(in_channels, out_channels, kernel_size, block)

    def forward(self, x):
        return self.sppf(x)


class SPPF(nn.Module):
    '''SPPF with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNSiLU):
        super().__init__()
        self.sppf = SPPFModule(in_channels, out_channels, kernel_size, block)

    def forward(self, x):
        return self.sppf(x)


class CSPSPPFModule(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5, block=ConvBNReLU):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = block(in_channels, c_, 1, 1)
        self.cv2 = block(in_channels, c_, 1, 1)
        self.cv3 = block(c_, c_, 3, 1)
        self.cv4 = block(c_, c_, 1, 1)

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = block(4 * c_, c_, 1, 1)
        self.cv6 = block(c_, c_, 3, 1)
        self.cv7 = block(2 * c_, out_channels, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x1)
            y2 = self.m(y1)
            y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        return self.cv7(torch.cat((y0, y3), dim=1))


class SimCSPSPPF(nn.Module):
    '''CSPSPPF with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5, block=ConvBNReLU):
        super().__init__()
        self.cspsppf = CSPSPPFModule(in_channels, out_channels, kernel_size, e, block)

    def forward(self, x):
        return self.cspsppf(x)


class CSPSPPF(nn.Module):
    '''CSPSPPF with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5, block=ConvBNSiLU):
        super().__init__()
        self.cspsppf = CSPSPPFModule(in_channels, out_channels, kernel_size, e, block)

    def forward(self, x):
        return self.cspsppf(x)


class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvModule_torch(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, activation_type=None, padding=padding, groups=groups)
            self.rbr_1x1 = ConvModule_torch(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, activation_type=None, padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.in_channels
        groups = self.groups
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule_torch):
            kernel = branch.conv.weight
            bias = branch.conv.bias
            return kernel, bias
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class QARepVGGBlock(RepVGGBlock):
    """
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://arxiv.org/abs/2212.01593
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(QARepVGGBlock, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        self._id_tensor = None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.bn(self.se(self.rbr_reparam(inputs))))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        bias = bias3x3

        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel = kernel + id_tensor
        return kernel, bias

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean - bias # remove bias
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        # keep post bn for QAT
        # if hasattr(self, 'bn'):
        #     self.__delattr__('bn')
        self.deploy = True


class QARepVGGBlockV2(RepVGGBlock):
    """
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://arxiv.org/abs/2212.01593
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(QARepVGGBlockV2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
            self.rbr_avg = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding) if out_channels == in_channels and stride == 1 else None
        self._id_tensor = None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.bn(self.se(self.rbr_reparam(inputs))))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.rbr_avg is None:
            avg_out = 0
        else:
            avg_out = self.rbr_avg(inputs)

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out + avg_out)))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        if self.rbr_avg is not None:
            kernelavg = self._avg_to_3x3_tensor(self.rbr_avg)
            kernel = kernel + kernelavg.to(self.rbr_1x1.weight.device)
        bias = bias3x3

        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel = kernel + id_tensor
        return kernel, bias

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean - bias # remove bias
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'rbr_avg'):
            self.__delattr__('rbr_avg')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        # keep post bn for QAT
        # if hasattr(self, 'bn'):
        #     self.__delattr__('bn')
        self.deploy = True


class RealVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, padding_mode='zeros', use_se=False,
    ):
        super(RealVGGBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

    def forward(self, inputs):
        out = self.relu(self.se(self.bn(self.conv(inputs))))
        return out


class ScaleLayer(torch.nn.Module):

    def __init__(self, num_features, use_bias=True, scale_init=1.0):
        super(ScaleLayer, self).__init__()
        self.weight = Parameter(torch.Tensor(num_features))
        init.constant_(self.weight, scale_init)
        self.num_features = num_features
        if use_bias:
            self.bias = Parameter(torch.Tensor(num_features))
            init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, inputs):
        if self.bias is None:
            return inputs * self.weight.view(1, self.num_features, 1, 1)
        else:
            return inputs * self.weight.view(1, self.num_features, 1, 1) + self.bias.view(1, self.num_features, 1, 1)


#   A CSLA block is a LinearAddBlock with is_csla=True
class LinearAddBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, padding_mode='zeros', use_se=False, is_csla=False, conv_scale_init=1.0):
        super(LinearAddBlock, self).__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.scale_conv = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=conv_scale_init)
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.scale_1x1 = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=conv_scale_init)
        if in_channels == out_channels and stride == 1:
            self.scale_identity = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=1.0)
        self.bn = nn.BatchNorm2d(out_channels)
        if is_csla:     # Make them constant
            self.scale_1x1.requires_grad_(False)
            self.scale_conv.requires_grad_(False)
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

    def forward(self, inputs):
        out = self.scale_conv(self.conv(inputs)) + self.scale_1x1(self.conv_1x1(inputs))
        if hasattr(self, 'scale_identity'):
            out += self.scale_identity(inputs)
        out = self.relu(self.se(self.bn(out)))
        return out



class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()

        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(*(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class BottleRep(nn.Module):

    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class BottleRep3(nn.Module):

    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        self.conv3 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class BepC3(nn.Module):
    '''CSPStackRep Block'''
    def __init__(self, in_channels, out_channels, n=1, e=0.5, block=RepVGGBlock):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv2 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv3 = ConvBNReLU(2 * c_, out_channels, 1, 1)
        if block == ConvBNSiLU:
            self.cv1 = ConvBNSiLU(in_channels, c_, 1, 1)
            self.cv2 = ConvBNSiLU(in_channels, c_, 1, 1)
            self.cv3 = ConvBNSiLU(2 * c_, out_channels, 1, 1)

        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BottleRep, basic_block=block)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class MBLABlock(nn.Module):
    ''' Multi Branch Layer Aggregation Block'''
    def __init__(self, in_channels, out_channels, n=1, e=0.5, block=RepVGGBlock):
        super().__init__()
        n = n // 2
        if n <= 0:
            n = 1

        # max add one branch
        if n == 1:
            n_list = [0, 1]
        else:
            extra_branch_steps = 1
            while extra_branch_steps * 2 < n:
                extra_branch_steps *= 2
            n_list = [0, extra_branch_steps, n]
        branch_num = len(n_list)

        c_ = int(out_channels * e)  # hidden channels
        self.c = c_
        self.cv1 = ConvModule_torch(in_channels, branch_num * self.c, 1, 1, 'relu', bias=False)
        self.cv2 = ConvModule_torch((sum(n_list) + branch_num) * self.c, out_channels, 1, 1,'relu', bias=False)

        if block == ConvBNSiLU:
            self.cv1 = ConvModule_torch(in_channels, branch_num * self.c, 1, 1, 'silu', bias=False)
            self.cv2 = ConvModule_torch((sum(n_list) + branch_num) * self.c, out_channels, 1, 1,'silu', bias=False)

        self.m = nn.ModuleList()
        for n_list_i in n_list[1:]:
            self.m.append(nn.Sequential(*(BottleRep3(self.c, self.c, basic_block=block, weight=True) for _ in range(n_list_i))))

        self.split_num = tuple([self.c]*branch_num)

    def forward(self, x):
        y = list(self.cv1(x).split(self.split_num, 1))
        all_y = [y[0]]
        for m_idx, m_i in enumerate(self.m):
            all_y.append(y[m_idx+1])
            all_y.extend(m(all_y[-1]) for m in m_i)
        return self.cv2(torch.cat(all_y, 1))


class BiFusion(nn.Module):
    '''BiFusion Block in PAN'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = ConvBNReLU(in_channels[0], out_channels, 1, 1)
        self.cv2 = ConvBNReLU(in_channels[1], out_channels, 1, 1)
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, 1, 1)

        self.upsample = Transpose(
            in_channels=out_channels,
            out_channels=out_channels,
        )
        self.downsample = ConvBNReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))


def get_block(mode):
    if mode == 'repvgg':
        return RepVGGBlock
    elif mode == 'qarepvgg':
        return QARepVGGBlock
    elif mode == 'qarepvggv2':
        return QARepVGGBlockV2
    elif mode == 'hyper_search':
        return LinearAddBlock
    elif mode == 'repopt':
        return RealVGGBlock
    elif mode == 'conv_relu':
        return ConvBNReLU
    elif mode == 'conv_silu':
        return ConvBNSiLU
    else:
        raise NotImplementedError("Undefied Repblock choice for mode {}".format(mode))


class SEBlock(nn.Module):

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        out = identity * x
        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class Lite_EffiBlockS1(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride):
        super().__init__()
        self.conv_pw_1 = ConvBNHS(
            in_channels=in_channels // 2,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        self.conv_dw_1 = ConvBN(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels)
        self.se = SEBlock(mid_channels)
        self.conv_1 = ConvBNHS(
            in_channels=mid_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
    def forward(self, inputs):
        x1, x2 = torch.split(
            inputs,
            split_size_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            dim=1)
        x2 = self.conv_pw_1(x2)
        x3 = self.conv_dw_1(x2)
        x3 = self.se(x3)
        x3 = self.conv_1(x3)
        out = torch.cat([x1, x3], axis=1)
        return channel_shuffle(out, 2)


class Lite_EffiBlockS2(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride):
        super().__init__()
        # branch1
        self.conv_dw_1 = ConvBN(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels)
        self.conv_1 = ConvBNHS(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        # branch2
        self.conv_pw_2 = ConvBNHS(
            in_channels=in_channels,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        self.conv_dw_2 = ConvBN(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2)
        self.se = SEBlock(mid_channels // 2)
        self.conv_2 = ConvBNHS(
            in_channels=mid_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        self.conv_dw_3 = ConvBNHS(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels)
        self.conv_pw_3 = ConvBNHS(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)

    def forward(self, inputs):
        x1 = self.conv_dw_1(inputs)
        x1 = self.conv_1(x1)
        x2 = self.conv_pw_2(inputs)
        x2 = self.conv_dw_2(x2)
        x2 = self.se(x2)
        x2 = self.conv_2(x2)
        out = torch.cat([x1, x2], axis=1)
        out = self.conv_dw_3(out)
        out = self.conv_pw_3(out)
        return out


class DPBlock(nn.Module):

    def __init__(self,
                 in_channel=96,
                 out_channel=96,
                 kernel_size=3,
                 stride=1):
        super().__init__()
        self.conv_dw_1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=out_channel,
            padding=(kernel_size - 1) // 2,
            stride=stride)
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.act_1 = nn.Hardswish()
        self.conv_pw_1 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=1,
            groups=1,
            padding=0)
        self.bn_2 = nn.BatchNorm2d(out_channel)
        self.act_2 = nn.Hardswish()

    def forward(self, x):
        x = self.act_1(self.bn_1(self.conv_dw_1(x)))
        x = self.act_2(self.bn_2(self.conv_pw_1(x)))
        return x

    def forward_fuse(self, x):
        x = self.act_1(self.conv_dw_1(x))
        x = self.act_2(self.conv_pw_1(x))
        return x


class DarknetBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv_1 = ConvBNHS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv_2 = DPBlock(
            in_channel=hidden_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=1)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        return out


class CSPBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 expand_ratio=0.5):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.conv_1 = ConvBNHS(in_channels, mid_channels, 1, 1, 0)
        self.conv_2 = ConvBNHS(in_channels, mid_channels, 1, 1, 0)
        self.conv_3 = ConvBNHS(2 * mid_channels, out_channels, 1, 1, 0)
        self.blocks = DarknetBlock(mid_channels,
                                   mid_channels,
                                   kernel_size,
                                   1.0)
    def forward(self, x):
        x_1 = self.conv_1(x)
        x_1 = self.blocks(x_1)
        x_2 = self.conv_2(x)
        x = torch.cat((x_1, x_2), axis=1)
        x = self.conv_3(x)
        return x



@MODELS.register_module()
class CSPRepBiFPANNeck(nn.Module):
    """
    CSPRepBiFPANNeck module.
    """

    def __init__(self,
                 scale_factor=0.75,
                 model_size='base',
                 freeze_all: bool = False,) -> None:
        super().__init__()

        channels_list=[64, 128, 256, 512, 1024, 256, 128, 128, 256, 256, 512]

        if model_size == 'small' or model_size == 'base' or model_size == 'large':
            num_repeats=[1, 6, 12, 18, 6, 12, 12, 12, 12] # small base large
        if model_size == 'tiny':
            num_repeats=[1, 6, 12, 18, 6, 6, 6, 6, 6] # tiny

        block=ConvBNSiLU
        csp_e=float(1)/2
        stage_block_type = "BepC3"

        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError
        
        scale_factor = scale_factor

        self.reduce_layer0 = ConvBNReLU(
            in_channels=int(channels_list[4] * scale_factor), # 1024
            out_channels=int(channels_list[5] * scale_factor), # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[int(channels_list[3] * scale_factor), int(channels_list[2] * scale_factor)], # 512, 256
            out_channels=int(channels_list[5] * scale_factor), # 256
        )

        self.Rep_p4 = stage_block(
            in_channels=int(channels_list[5] * scale_factor), # 256
            out_channels=int(channels_list[5] * scale_factor), # 256
            n=num_repeats[5],
            e=csp_e,
            block=block
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=int(channels_list[5] * scale_factor), # 256
            out_channels=int(channels_list[6] * scale_factor), # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[int(channels_list[2] * scale_factor), int(channels_list[1] * scale_factor)], # 256, 128
            out_channels=int(channels_list[6] * scale_factor), # 128
        )

        self.Rep_p3 = stage_block(
            in_channels=int(channels_list[6] * scale_factor), # 128
            out_channels=int(channels_list[6] * scale_factor), # 128
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        self.downsample2 = ConvBNReLU(
            in_channels=int(channels_list[6] * scale_factor), # 128
            out_channels=int(channels_list[7] * scale_factor), # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = stage_block(
            in_channels=int(channels_list[6] * scale_factor) + int(channels_list[7] * scale_factor), # 128 + 128
            out_channels=int(channels_list[8] * scale_factor), # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        self.downsample1 = ConvBNReLU(
            in_channels=int(channels_list[8] * scale_factor), # 256
            out_channels=int(channels_list[9] * scale_factor), # 256
            kernel_size=3,
            stride=2
        )


        self.Rep_n4 = stage_block(
            in_channels=int(channels_list[5] * scale_factor) + int(channels_list[9] * scale_factor), # 256 + 256
            out_channels=int(channels_list[10] * scale_factor), # 512
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

        self.freeze_all = freeze_all
        if self.freeze_all:
            self._freeze_all()


    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()
            
    
    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]


        return outputs





@MODELS.register_module()
class YOLOWorldPAFPN(YOLOv8PAFPN):
    """Path Aggregation Network used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 block_cfg: ConfigType = dict(type='CSPLayerWithTwoConv'),
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.block_cfg = block_cfg
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.in_channels[idx - 1] + self.in_channels[idx]),
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx - 1],
                                             self.widen_factor),
                 guide_channels=self.guide_channels,
                 embed_channels=make_round(self.embed_channels[idx - 1],
                                           self.widen_factor),
                 num_heads=make_round(self.num_heads[idx - 1],
                                      self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx + 1],
                                             self.widen_factor),
                 guide_channels=self.guide_channels,
                 embed_channels=make_round(self.embed_channels[idx + 1],
                                           self.widen_factor),
                 num_heads=make_round(self.num_heads[idx + 1],
                                      self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))


        return tuple(results)


@MODELS.register_module()
class YOLOWorldDualPAFPN(YOLOWorldPAFPN):
    """Path Aggregation Network used in YOLO World v8."""
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 text_enhancder: ConfigType = dict(
                     type='ImagePoolingAttentionModule',
                     embed_channels=256,
                     num_heads=8,
                     pool_size=3),
                 block_cfg: ConfigType = dict(type='CSPLayerWithTwoConv'),
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         guide_channels=guide_channels,
                         embed_channels=embed_channels,
                         num_heads=num_heads,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         block_cfg=block_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        text_enhancder.update(
            dict(
                image_channels=[int(x * widen_factor) for x in out_channels],
                text_channels=guide_channels,
                num_feats=len(out_channels),
            ))
        print(text_enhancder)
        self.text_enhancer = MODELS.build(text_enhancder)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor) -> tuple:
        """Forward function."""
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        txt_feats = self.text_enhancer(txt_feats, inner_outs)
        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)
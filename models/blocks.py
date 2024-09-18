import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


##############################################################################################
#                                                                                            #
#  ERFNET blocks from https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py  #
#                                                                                            #
##############################################################################################

class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated, batch_norm=False, instance_norm=False, init=None):
        super().__init__()
        self.batch_norm = batch_norm
        self.instance_norm = instance_norm
        self.init = init

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        if self.instance_norm:
            self.in1_ = torch.nn.InstanceNorm2d(chann)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1 * dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1 * dilated), bias=True, dilation=(1, dilated))

        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        if self.instance_norm:
            self.in2_ = torch.nn.InstanceNorm2d(chann)

        self.dropout = nn.Dropout2d(dropprob)

        # initialization
        if self.init == 'he':
            nn.init.kaiming_normal_(self.conv1x3_1.weight, mode='fan_out', nonlinearity='gelu')
            nn.init.kaiming_normal_(self.conv1x3_1.weight, mode='fan_out', nonlinearity='gelu')
            nn.init.kaiming_normal_(self.conv3x1_2.weight, mode='fan_out', nonlinearity='gelu')
            nn.init.kaiming_normal_(self.conv1x3_2.weight, mode='fan_out', nonlinearity='gelu')
            if self.batch_norm:
                nn.init.constant_(self.bn1.weight, 1)
                nn.init.constant_(self.bn1.bias, 0)
                nn.init.constant_(self.bn2.weight, 1)
                nn.init.constant_(self.bn2.bias, 0)
        elif self.init == 'xavier':
            nn.init.xavier_normal_(self.conv1x3_1.weight)
            nn.init.xavier_normal_(self.conv1x3_1.weight)
            nn.init.xavier_normal_(self.conv3x1_2.weight)
            nn.init.xavier_normal_(self.conv1x3_2.weight)
            if self.batch_norm:
                nn.init.constant_(self.bn1.weight, 1)
                nn.init.constant_(self.bn1.bias, 0)
                nn.init.constant_(self.bn2.weight, 1)
                nn.init.constant_(self.bn2.bias, 0)
        elif self.init != "None":
            raise AttributeError("Invalid initialization")

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.gelu(output)
        output = self.conv1x3_1(output)
        if self.batch_norm:
            output = self.bn1(output)
        if self.instance_norm:
            output = self.in1_(output)
        output = F.gelu(output)

        output = self.conv3x1_2(output)
        output = F.gelu(output)
        output = self.conv1x3_2(output)
        if self.batch_norm:
            output = self.bn2(output)
        if self.instance_norm:
            output = self.in2_(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.gelu(output + input)  # +input = identity (residual connection)


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, batch_norm=False, instance_norm=False, init=None):
        super().__init__()
        self.init = init
        self.conv = nn.Conv2d(ninput, noutput - ninput,
                              (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.batch_norm = batch_norm
        self.instance_norm = instance_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        if self.instance_norm:
            self.in_ = torch.nn.InstanceNorm2d(noutput)

        # initialization
        if self.init == 'he':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='gelu')
            if self.batch_norm:
                nn.init.constant_(self.bn.weight, 1)
                nn.init.constant_(self.bn.bias, 0)
        elif self.init == 'xavier':
            nn.init.xavier_normal_(self.conv.weight)
            if self.batch_norm:
                nn.init.constant_(self.bn.weight, 1)
                nn.init.constant_(self.bn.bias, 0)
        elif self.init != "None":
            raise AttributeError("Invalid initialization")

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        if self.batch_norm:
            output = self.bn(output)
        if self.instance_norm:
            output = self.in_(output)
        return F.gelu(output)


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, init=None):
        super().__init__()
        self.init = init

        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

        # initialization
        if self.init == 'he':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='gelu')
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        elif self.init == 'xavier':
            nn.init.xavier_normal_(self.conv.weight)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        elif self.init != "None":
            raise AttributeError("Invalid initialization")

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.gelu(output)


class ConvBNAct(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size,
                 activation=nn.ReLU(inplace=True), dilation=1, stride=1):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride))
        self.add_module('bn', nn.BatchNorm2d(channels_out))
        self.add_module('act', activation)


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            # nn.Dropout2d(0.2),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)
        # self.fusion_conv = nn.Conv2d(channels_in, channels_in, kernel_size=1)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        # out = self.fusion_conv(out)
        return (out)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, out_dim, bins=(1, 2, 3, 6),
                 activation=nn.ReLU(inplace=True),
                 upsampling_mode='bilinear'):
        reduction_dim = in_dim // len(bins)
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        self.upsampling_mode = upsampling_mode
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                ConvBNAct(in_dim, reduction_dim, kernel_size=1,
                          activation=activation)
            ))
        in_dim_last_conv = in_dim + reduction_dim * len(bins)
        self.features = nn.ModuleList(self.features)

        self.final_conv = ConvBNAct(in_dim_last_conv, out_dim,
                                    kernel_size=1, activation=activation)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            h, w = x_size[2:]
            y = f(x)
            if self.upsampling_mode == 'nearest':
                out.append(F.interpolate(y, (int(h), int(w)), mode='nearest'))
            elif self.upsampling_mode == 'bilinear':
                out.append(F.interpolate(y, (int(h), int(w)),
                                         mode='bilinear',
                                         align_corners=False))
            else:
                raise NotImplementedError(
                    'For the PyramidPoolingModule only nearest and bilinear '
                    'interpolation are supported. '
                    f'Got: {self.upsampling_mode}'
                )
        out = torch.cat(out, 1)
        out = self.final_conv(out)
        return out


class AdaptivePyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, out_dim, input_size, bins=(1, 2, 3, 6),
                 activation=nn.ReLU(inplace=True), upsampling_mode='bilinear'):
        reduction_dim = in_dim // len(bins)
        super(AdaptivePyramidPoolingModule, self).__init__()
        self.features = []
        self.upsampling_mode = upsampling_mode
        self.input_size = input_size
        self.bins = bins
        for _ in bins:
            self.features.append(
                ConvBNAct(in_dim, reduction_dim, kernel_size=1,
                          activation=activation)
            )
        in_dim_last_conv = in_dim + reduction_dim * len(bins)
        self.features = nn.ModuleList(self.features)

        self.final_conv = ConvBNAct(in_dim_last_conv, out_dim,
                                    kernel_size=1, activation=activation)

    def forward(self, x):
        x_size = x.size()
        h, w = x_size[2:]
        h_inp, w_inp = self.input_size
        bin_multiplier_h = int((h / h_inp) + 0.5)
        bin_multiplier_w = int((w / w_inp) + 0.5)
        out = [x]
        for f, bin in zip(self.features, self.bins):
            h_pool = bin * bin_multiplier_h
            w_pool = bin * bin_multiplier_w
            pooled = F.adaptive_avg_pool2d(x, (h_pool, w_pool))
            y = f(pooled)
            if self.upsampling_mode == 'nearest':
                out.append(F.interpolate(y, (int(h), int(w)), mode='nearest'))
            elif self.upsampling_mode == 'bilinear':
                out.append(F.interpolate(y, (int(h), int(w)),
                                         mode='bilinear',
                                         align_corners=False))
            else:
                raise NotImplementedError(
                    'For the PyramidPoolingModule only nearest and bilinear '
                    'interpolation are supported. '
                    f'Got: {self.upsampling_mode}'
                )
        out = torch.cat(out, 1)
        out = self.final_conv(out)
        return out

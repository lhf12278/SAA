import torch
import torch.nn as nn



def conv1x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,3), stride=stride,
                     padding=(0,1), groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv2 = conv1x3(inplanes, planes, stride, groups, dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x3(planes, planes, stride, groups, dilation)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x



        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ft(nn.Module):

    def __init__(self, args, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ft, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if args.embedding_type == 'BERT':
            self.inplanes = 768
        else:
            self.inplanes=args.embedding_size
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        downsample = nn.Sequential(
            conv1x1(1024, 2048),
            norm_layer(2048),
        )

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))


        self.conv1 = conv1x1(self.inplanes, 1024)
        self.bn1 = norm_layer(1024)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(1024, 2048)

        self.bn1 = norm_layer(1024)
        self.bn2=nn.BatchNorm2d(2048)
        self.bn3=nn.BatchNorm1d(2048)
        self.relu = nn.ReLU(inplace=True)

        self.branch1 = nn.Sequential(
            Bottleneck(inplanes=1024, planes=2048, width=512, downsample=downsample),

            Bottleneck(inplanes=2048, planes=2048, width=512)
        )
        self.branch2 = nn.Sequential(
            Bottleneck(inplanes=1024, planes=2048, width=512, downsample=downsample),

            Bottleneck(inplanes=2048, planes=2048, width=512)
        )
        self.branch3 = nn.Sequential(
            Bottleneck(inplanes=1024, planes=2048, width=512, downsample=downsample),

            Bottleneck(inplanes=2048, planes=2048, width=512)
        )
        self.branch4 = nn.Sequential(
            Bottleneck(inplanes=1024, planes=2048, width=512, downsample=downsample),

            Bottleneck(inplanes=2048, planes=2048, width=512)
        )
        self.branch5 = nn.Sequential(
            Bottleneck(inplanes=1024, planes=2048, width=512, downsample=downsample),

            Bottleneck(inplanes=2048, planes=2048, width=512)
        )
        self.branch6 = nn.Sequential(
            Bottleneck(inplanes=1024, planes=2048, width=512, downsample=downsample),

            Bottleneck(inplanes=2048, planes=2048, width=512)
        )
        self.branch7 = nn.Sequential(
            Bottleneck(inplanes=1024, planes=2048, width=512, downsample=downsample),

            Bottleneck(inplanes=2048, planes=2048, width=512)
        )
        self.branch8 = nn.Sequential(
            Bottleneck(inplanes=1024, planes=2048, width=512, downsample=downsample),

            Bottleneck(inplanes=2048, planes=2048, width=512)
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)


    def forward(self, x):
        x1 = self.conv1(x)  # 1024 1 64
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x11 = self.max_pool(x1).squeeze(dim=-1).squeeze(dim=-1)

        x21 = self.branch1(x1)
        x22 = self.branch2(x1)
        x23 = self.branch3(x1)
        # x24 = self.branch4(x1)
        # x25 = self.branch5(x1)
        # x26 = self.branch6(x1)
        # x27 = self.branch7(x1)
        # x28 = self.branch8(x1)

        return x11, x21, x22, x23

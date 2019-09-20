import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': '/raid/lcq/premdoel_pytorch/resnet18-5c106cde.pth',
    'resnet34': '/raid/lcq/premdoel_pytorch/resnet34-333f7ec4.pth',
    'resnet50': '/raid/lcq/premdoel_pytorch/resnet50-19c8e357.pth',
    'resnet101': '/raid/lcq/premdoel_pytorch/resnet101-5d3b4d8f.pth',
    'resnet152': '/raid/lcq/premdoel_pytorch/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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

        out += residual
        out = self.relu(out)

        return out

class ASPP(nn.Module):
    def __init__(self,in_plant,out_plant):
        super(ASPP,self).__init__()
        self.conv1x1=nn.Sequential(
            nn.Conv2d(in_plant,96,1,1,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        self.con3x3_6=nn.Sequential(
            nn.Conv2d(in_plant,96,3,1,padding=6,dilation=6,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_12=nn.Sequential(
            nn.Conv2d(in_plant, 96, 3, 1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )


        self.conv3x3_18=nn.Sequential(
            nn.Conv2d(in_plant, 96, 3, 1, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        self.pool=nn.MaxPool2d(2,2)

        self.com=nn.Sequential(
            nn.Conv2d(512,out_plant,1,bias=False),
            nn.BatchNorm2d(out_plant),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_8,x_16):
        out1=self.conv1x1(x_16)
        out2=self.con3x3_6(x_16)
        out3=self.conv3x3_12(x_16)
        out4=self.conv3x3_18(x_16)
        out5=self.pool(x_8)
        out=torch.cat([out1,out2,out3,out4,out5],1)
        out=self.com(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # /2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)  # /2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # /2(28x28)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # /2(14x14)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)  # /2

        self.atrou=nn.Sequential(
            nn.Conv2d(512,512,3,1,padding=2,dilation=2,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.aspp=ASPP(512,512)
        self.deconvs=nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,num_classes,1,1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride==1and self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            elif stride!=1 and self.inplanes == planes * block.expansion:
                downsample = nn.AvgPool2d(2,2)

            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    nn.AvgPool2d(2, 2)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_8 = x
        x = self.layer3(x)
        x = self.layer4(x)
        x_16=x
        x=self.aspp(x_8,x_16)
        x=self.deconvs(x)
        return x

    def load_weight(self, path):
        state_dict = torch.load(path)
        temp_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.load_state_dict(temp_dict)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        static_dict = torch.load(model_urls['resnet18'])
        temp_dict = model.state_dict().keys()
        temp_k = ['fc.weight', 'fc.bias']
        temp_dict = {k: v for k, v in static_dict.items() if k in temp_dict and (k not in temp_k)}
        state_dict = model.state_dict()
        state_dict.update(temp_dict)
        model.load_state_dict(state_dict)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        static_dict = torch.load(model_urls['resnet34'])
        temp_dict = model.state_dict().keys()
        temp_k = ['fc.weight', 'fc.bias']
        temp_dict = {k: v for k, v in static_dict.items() if k in temp_dict and (k not in temp_k)}
        state_dict = model.state_dict()
        state_dict.update(temp_dict)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        static_dict = torch.load(model_urls['resnet50'])
        temp_dict = model.state_dict().keys()
        temp_k = ['fc.weight', 'fc.bias']
        temp_dict = {k: v for k, v in static_dict.items() if k in temp_dict and (k not in temp_k)}
        state_dict = model.state_dict()
        state_dict.update(temp_dict)
        model.load_state_dict(state_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = torch.load(model_urls['resnet101'])
        temp_keys = model.state_dict().keys()
        temp_dict = {k: v for k, v in state_dict.items() if k in temp_keys}
        state_dict = model.state_dict()
        state_dict.update(temp_dict)
        model.load_state_dict(state_dict)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        state_dict = torch.load(model_urls['resnet152'])
        temp_keys = model.state_dict().keys()
        temp_dict = {k: v for k, v in state_dict.items() if k in temp_keys}
        state_dict = model.state_dict()
        state_dict.update(temp_dict)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    pass
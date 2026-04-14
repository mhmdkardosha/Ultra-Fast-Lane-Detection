import torch,pdb
import torchvision
import torch.nn.modules


def _build_model(builder, weight_enum_name, pretrained):
    if pretrained:
        try:
            weights_enum = getattr(torchvision.models, weight_enum_name)
            return builder(weights=weights_enum.DEFAULT)
        except (AttributeError, TypeError):
            return builder(pretrained=True)
    try:
        return builder(weights=None)
    except TypeError:
        return builder(pretrained=False)

class vgg16bn(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(vgg16bn,self).__init__()
        model = list(_build_model(torchvision.models.vgg16_bn, 'VGG16_BN_Weights', pretrained).features.children())
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)
        
    def forward(self,x):
        return self.model(x)
class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            model = _build_model(torchvision.models.resnet18, 'ResNet18_Weights', pretrained)
        elif layers == '34':
            model = _build_model(torchvision.models.resnet34, 'ResNet34_Weights', pretrained)
        elif layers == '50':
            model = _build_model(torchvision.models.resnet50, 'ResNet50_Weights', pretrained)
        elif layers == '101':
            model = _build_model(torchvision.models.resnet101, 'ResNet101_Weights', pretrained)
        elif layers == '152':
            model = _build_model(torchvision.models.resnet152, 'ResNet152_Weights', pretrained)
        elif layers == '50next':
            model = _build_model(torchvision.models.resnext50_32x4d, 'ResNeXt50_32X4D_Weights', pretrained)
        elif layers == '101next':
            model = _build_model(torchvision.models.resnext101_32x8d, 'ResNeXt101_32X8D_Weights', pretrained)
        elif layers == '50wide':
            model = _build_model(torchvision.models.wide_resnet50_2, 'Wide_ResNet50_2_Weights', pretrained)
        elif layers == '101wide':
            model = _build_model(torchvision.models.wide_resnet101_2, 'Wide_ResNet101_2_Weights', pretrained)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2,x3,x4

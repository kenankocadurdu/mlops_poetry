import torch
import torchvision
import torch.nn.functional as F
from utils import model_arch
import logging
from fastai.vision.all import *

class generator:
    def __init__(self, name, num_class, image_size):
        self.name = name
        self.num_class = num_class
        self.image_size = image_size

        if self.name == "mobilenet_v2":
            self.model = mobilenet_v2(self.num_class) 
        elif self.name == "mobilenet_v3":
            self.model = mobilenet_v3(self.num_class)  
        elif self.name == "MNASNet":
            self.model = MNASNet(self.num_class)  
        elif self.name == "vgg19":
            self.model = vgg19(self.num_class)  
        elif self.name == "densenet201":
            self.model = densenet201(self.num_class) 
        elif self.name == "resnet50":
            self.model = resnet50(self.num_class)  
        elif self.name == "resnext50_32x4d":
            self.model = resnext50_32x4d(self.num_class)
        elif self.name == "XResnet50":
            self.model = XResnet50(self.num_class)
        elif self.name == "convnext_large":
            self.model = convnext_large(self.num_class)
        elif self.name == "XNet":
            self.model = XNet(num_class = self.num_class)
        elif self.name == "EModel":
            self.model = EModel(self.image_size)

        logging.info("-"*50)
        logging.info("Model generated...")
        logging.info("model name : {}".format(self.name))
        logging.info("num class : {}".format(self.num_class))
        logging.info("image size : {}".format(self.image_size))


##################### standarts ##############################
class mobilenet_v2(model_arch.Module):
    def __init__(self, num_class):
        super().__init__()
        self.network = torchvision.models.mobilenet_v2(weights='DEFAULT')
        for param in self.network.parameters():
            param.requires_grad = False
        self.network.classifier[-1] = torch.nn.Linear(in_features=1280, out_features=num_class, bias=True)

    def forward(self, xb):
        return self.network.forward(xb)

class mobilenet_v3(model_arch.Module):
    def __init__(self, num_class):
        super().__init__()
        self.network = torchvision.models.mobilenet_v3_large(weights=None)
        self.network.classifier[3] = torch.nn.Linear(in_features=1280, out_features=num_class, bias=True)
    
    def forward(self, xb):
        return self.network.forward(xb)

class MNASNet(model_arch.Module):
    def __init__(self, num_class):
        super().__init__()
        self.network = torchvision.models.MNASNet(alpha=1)
        self.network.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_class, bias=True)
    
    def forward(self, xb):
        return self.network.forward(xb)

class densenet201(model_arch.Module):
    def __init__(self, num_class):
        super().__init__()
        self.network = torchvision.models.densenet201(weights=None)
        self.network.classifier = torch.nn.Linear(in_features=1920, out_features=num_class, bias=True)

    def forward(self, xb):
        return self.network.forward(xb)

class convnext_large(model_arch.Module):
    def __init__(self, num_class):
        super().__init__()
        self.network = torchvision.models.convnext_large()
        self.network.classifier[-1] = torch.nn.Linear(in_features=1536, out_features=num_class, bias=True)
    
    def forward(self, xb):
        return self.network.forward(xb)

class vgg19(model_arch.Module):
    def __init__(self, num_class):
        super().__init__()
        self.network = torchvision.models.vgg19(weights=None)
        self.network.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_class, bias=True)
    
        for p in self.network.features.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network.features(x)
        x = self.network.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.network.classifier(x)
        return x

class resnet50(model_arch.Module):
    def __init__(self, num_class):
        super().__init__()
        self.network = torchvision.models.resnet50(weights=None)
        self.network.fc = torch.nn.Linear(in_features=2048, out_features=num_class, bias=True)
    
    def forward(self, xb):
        return self.network.forward(xb)

class resnext50_32x4d(model_arch.Module):
    def __init__(self, num_class):
        super().__init__()
        self.network = torchvision.models.resnext50_32x4d(weights=None)
        self.network.fc = torch.nn.Linear(in_features=2048, out_features=num_class, bias=True)
    
    def forward(self, xb):
        return self.network.forward(xb)

class XResnet50(model_arch.Module):
    def __init__(self, num_class):
        super().__init__()
        self.network = xresnet50(pretrained=True)
        self.network[-1] = torch.nn.Linear(in_features=2048, out_features=num_class, bias=True)
    
    def forward(self, xb):
        return self.network.forward(xb)

##################### Ensembles ##############################
class EModel(model_arch.Module):
    def __init__(self, num_class=2):
        super(EModel, self).__init__()
        
        net1 = torchvision.models.densenet201(weights='DEFAULT')
        #net1.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features_net1 = nn.Sequential(*list(net1.children())[:-1]) # 1920

        net2 = torchvision.models.mobilenet_v3_large(weights='DEFAULT')
        #net2.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.features_net2 = net2.features # 960

        #net1 = torchvision.models.resnet50(weights='DEFAULT')
        #self.features_net1 = nn.Sequential(*list(net1.children())[:-2])  # 2048

        #net3 = xresnet50(pretrained=True)
        #net3[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #self.features_net3 = nn.Sequential(*list(net3.children())[:-4]) # 2048

        #net2 = torchvision.models.resnext50_32x4d(weights='DEFAULT')
        #self.features_net2 = nn.Sequential(*list(net2.children())[:-2])  # 2048

        #net2 = XNet()
        #self.features_net2 = net2.features # 64


        self.classifier = nn.Sequential(
            nn.Linear((1920 + 960) * 9, 2048),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.1),
            nn.Linear(2048, num_class)
        )

    def forward(self, x):
        x_net1 = self.features_net1(x)
        x_net1 = F.adaptive_avg_pool2d(x_net1, (3, 3)).reshape(x_net1.size(0), -1)
        x_net2 = self.features_net2(x)
        x_net2 = F.adaptive_avg_pool2d(x_net2, (3, 3)).reshape(x_net2.size(0), -1)
        #x_net3 = self.features_net3(x)
        #x_net3 = F.adaptive_avg_pool2d(x_net3, (1, 1)).reshape(x_net3.size(0), -1)
        #combined_features = torch.cat((x_net1, x_net2, x_net3), dim=1)
        combined_features = torch.cat((x_net1, x_net2), dim=1)
        
        output = self.classifier(combined_features)
        return output

##################### MyNetworks ##############################
act_func = nn.SiLU()
# nn.Tanh() - nn.Softsign()  - nn.Mish() - nn.SiLU() - nn.GELU() - nn.CELU() - 
# nn.SELU() - nn.RReLU(0.1, 0.3) - nn.PReLU() - nn.ReLU6(inplace=True)

class Connection1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Connection1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = act_func

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.act1(x)
        return x

class Connection3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Connection3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = act_func

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.act1(x)
        return x
    
class Connection5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Connection5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = act_func

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.act1(x)
        return x


class XNet(model_arch.Module):
    def __init__(self, num_class=2):
        super(XNet, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            act_func,

            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            act_func,

            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            act_func,

        )

        self.connection1 = Connection1(128, 128)
        self.connection3 = Connection3(128, 128)
        self.connection5 = Connection5(128, 128)
        
        self.features2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            act_func,
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 4096),
            act_func,
            torch.nn.Linear(4096, 2048),
            act_func,
            torch.nn.Linear(2048, 1024),
            act_func,
            torch.nn.Linear(1024, 512),

            nn.Dropout(0.25),
            act_func,
            nn.Linear(512, num_class),
        )
        #nn.init.kaiming_uniform_

        nn.init.xavier_uniform_(self.features1[0].weight)
        nn.init.xavier_uniform_(self.features1[3].weight)
        nn.init.xavier_uniform_(self.features1[6].weight)
        nn.init.xavier_uniform_(self.connection1.conv1.weight)
        nn.init.xavier_uniform_(self.connection3.conv1.weight)
        nn.init.xavier_uniform_(self.connection5.conv1.weight)

        nn.init.xavier_uniform_(self.features2[0].weight)

    
    def forward(self, x):
        x1 = self.features1(x)
        c1 = self.connection1(x1)
        c3 = self.connection3(x1)
        c5 = self.connection5(x1)

        x = c1 + c3 + c5
        x = self.features2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

import torch
from torch import nn
from torch.nn import functional as F
from .point_pillar import PointPillarNet



class LiDARModel(nn.Module):
    def __init__(self, num_input=9, num_features=[32,32],
        backbone='swin',
        min_x=-10, max_x=70,
        min_y=-40, max_y=40,
        pixels_per_meter=4):

        super().__init__()

        self.point_pillar_net = PointPillarNet(
            num_input, num_features,
            min_x=min_x, max_x=max_x,
            min_y=min_y, max_y=max_y,
            pixels_per_meter=pixels_per_meter,
        )

        num_feature = num_features[-1]
        if backbone == 'cnn':
            self.backbone = ConvBackbone(num_feature=num_feature)
        else:
            raise NotImplementedError

        self.center_head = Head(6*num_feature,2) 
        self.box_head    = Head(6*num_feature,2)
        self.ori_head    = Head(6*num_feature,2)
        self.seg_head    = Head(6*num_feature,3, output_activation=torch.sigmoid)

    def forward(self, lidars, num_points):

        features = self.point_pillar_net(lidars, num_points)
        features = self.backbone(features)

        return (features,
            self.center_head(features),
            self.box_head(features),
            self.ori_head(features),
            self.seg_head(features),
        )


class ConvBackbone(nn.Module):
    def __init__(self, num_feature=64, norm_cfg={'eps': 1e-3, 'momentum': 0.01}):
        """
        Original PointPillar Backbone
        TODO: Write this better...
        """

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_feature,num_feature,3,2,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_feature, **norm_cfg),
            nn.Conv2d(num_feature,num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_feature, **norm_cfg),
            nn.Conv2d(num_feature,num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_feature, **norm_cfg),
            nn.Conv2d(num_feature,num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_feature, **norm_cfg),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_feature,2*num_feature,3,2,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(2*num_feature,2*num_feature,3,2,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
        )

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(num_feature,2*num_feature,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),            
        )

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(2*num_feature,2*num_feature,4,2,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),            

        )

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(2*num_feature,2*num_feature,4,4,1,2,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),            
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        u1 = self.upconv1(x1)
        u2 = self.upconv2(x2)
        u3 = self.upconv3(x3)

        return torch.cat([u1,u2,u3], dim=1)



class Head(nn.Module):
    def __init__(self, num_input, num_output, num_hidden=64, norm_cfg={'eps': 1e-3, 'momentum': 0.01}, output_activation=nn.Identity()):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_input,num_hidden,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_hidden, **norm_cfg),
            nn.ConvTranspose2d(num_hidden,num_output,3,2,1,1),
        )
        self.output_activation = output_activation

    def forward(self, x):
        out = self.net(x)
        return self.output_activation(out)

if __name__ == '__main__':
    
    swin = ConvBackbone()
    print (swin(torch.zeros((2,64,320,320))).shape)

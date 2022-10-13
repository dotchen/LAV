import torch
from torch import nn
from torch.nn import functional as F
from common.resnet import resnet18, resnet34
from common.swin_transformer import ClassifySwin
from common.normalize import Normalize
from common.segmentation import SegmentationHead


class CameraModelV3(nn.Module):
    def __init__(self, config, num_cmds=6):
        super().__init__()

        # Configs
        self.num_cmds   = num_cmds
        self.num_steers = config['num_steers']
        self.num_throts = config['num_throts']
        self.num_speeds = config['num_speeds']
        self.num_labels = len(config['seg_channels'])
        self.all_speeds = config['all_speeds']
        self.two_cam    = config['use_narr_cam']

        self.seg_channels = config['seg_channels']

        self.seg_model = nn.Sequential(
            resnet18(pretrained=config['imagenet_pretrained']),
            SegmentationHead(512, self.num_labels+1),
        )

        self.act_model = nn.Sequential(
            SwinTransformer(),
            
        )
        
        self.bra_model = nn.Sequential(
            
        )

class CameraModelV2(nn.Module):
    def __init__(self, config, num_cmds=6):
        super().__init__()

        # Configs
        self.num_cmds   = num_cmds
        self.num_steers = config['num_steers']
        self.num_throts = config['num_throts']
        self.num_speeds = config['num_speeds']
        self.num_labels = len(config['seg_channels'])
        self.all_speeds = config['all_speeds']
        self.two_cam    = config['use_narr_cam']
        
        self.seg_channels = config['seg_channels']

        self.seg_wide = nn.Sequential(
            resnet18(pretrained=config['imagenet_pretrained']),
            SegmentationHead(512, self.num_labels+1)
        )
        self.act_back_wide = nn.Sequential(
            resnet34(pretrained=False, num_channels=4+len(self.seg_channels)),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        if self.two_cam:
            self.seg_narr = nn.Sequential(
                resnet18(pretrained=config['imagenet_pretrained']),
                SegmentationHead(512, self.num_labels+1)
            )
            self.act_back_narr = nn.Sequential(
                resnet18(pretrained=False, num_channels=4+len(self.seg_channels)),
                nn.Conv2d(512,64,1,1),
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
            )
        
        if self.all_speeds:
            self.num_acts = self.num_cmds*self.num_speeds*(self.num_steers+self.num_throts+1)
        else:
            self.num_acts = self.num_cmds*(self.num_steers+self.num_throts+1)
            self.spd_encoder = nn.Sequential(
                nn.Linear(1,64),
                nn.ReLU(True),
                nn.Linear(64,64),
                nn.ReLU(True),
            )
    
        self.act_head = nn.Linear(512 + (0 if self.all_speeds else 64) + (64 if self.two_cam else 0), self.num_acts)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, wide_rgb, narr_rgb, spd=None):
        
        assert (self.all_speeds and spd is None) or \
               (not self.all_speeds and spd is not None)

        wide_rgb = self.normalize(wide_rgb/255.)
        wide_seg = F.interpolate(self.seg_wide(wide_rgb), scale_factor=4)
        wide_cls = F.one_hot(wide_seg.detach().argmax(dim=1), num_classes=len(self.seg_channels)+1).float().permute(0,3,1,2)
        wide_msk = (wide_cls.argmax(dim=1, keepdim=True) > 0).float()

        if self.two_cam:
            narr_rgb = self.normalize(narr_rgb/255.)
            narr_seg = F.interpolate(self.seg_narr(narr_rgb), scale_factor=4)
            narr_cls = F.one_hot(narr_seg.detach().argmax(dim=1), num_classes=len(self.seg_channels)+1).float().permute(0,3,1,2)
            narr_msk = (narr_cls.argmax(dim=1, keepdim=True) > 0).float()

            embed = torch.cat([
                self.act_back_wide(torch.cat([wide_rgb*wide_msk, wide_cls], dim=1)),
                self.act_back_narr(torch.cat([narr_rgb*narr_msk, narr_cls], dim=1))
            ], dim=1)
        else:
            embed = self.act_back_wide(wide_rgb*wide_msk)

        # Action logits
        if self.all_speeds:
            act_output = self.act_head(embed).view(-1,self.num_cmds,self.num_speeds,self.num_steers+self.num_throts+1)
            act_output = action_logits(act_output, self.num_steers, self.num_throts)
        else:
            act_output = self.act_head(torch.cat([embed, self.spd_encoder(spd[:,None])], dim=1)).view(-1,self.num_cmds,1,self.num_steers+self.num_throts+1)
            act_output = action_logits(act_output, self.num_steers, self.num_throts).squeeze(2)

        if self.two_cam:
            return act_output, wide_seg, narr_seg
        else:
            return act_output, wide_seg
        
class CameraModel(nn.Module):
    def __init__(self, config, num_cmds=6):
        super().__init__()

        # Configs
        self.num_cmds   = num_cmds
        self.num_steers = config['num_steers']
        self.num_throts = config['num_throts']
        self.num_speeds = config['num_speeds']
        self.num_labels = len(config['seg_channels'])
        self.all_speeds = config['all_speeds']
        self.two_cam    = config['use_narr_cam']
        
        self.backbone_wide = resnet34(pretrained=config['imagenet_pretrained'])
        self.seg_head_wide = SegmentationHead(512, self.num_labels+1)
        if self.two_cam:
            self.backbone_narr = resnet18(pretrained=config['imagenet_pretrained'])
            self.seg_head_narr = SegmentationHead(512, self.num_labels+1)
            self.bottleneck_narr = nn.Sequential(
                nn.Linear(512,64),
                nn.ReLU(True),
            )

        if self.all_speeds:
            self.num_acts = self.num_cmds*self.num_speeds*(self.num_steers+self.num_throts+1)
        else:
            self.num_acts = self.num_cmds*(self.num_steers+self.num_throts+1)
            self.spd_encoder = nn.Sequential(
                nn.Linear(1,64),
                nn.ReLU(True),
                nn.Linear(64,64),
                nn.ReLU(True),
            )

        self.wide_seg_head = SegmentationHead(512, self.num_labels+1)
        self.act_head = nn.Sequential(
            nn.Linear(512 + (0 if self.all_speeds else 64) + (64 if self.two_cam else 0),256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,self.num_acts),
        )
        
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, wide_rgb, narr_rgb, spd=None):
        
        assert (self.all_speeds and spd is None) or \
               (not self.all_speeds and spd is not None)

        wide_embed = self.backbone_wide(self.normalize(wide_rgb/255.))
        wide_seg_output = self.seg_head_wide(wide_embed)

        if self.two_cam:
            narr_embed = self.backbone_narr(self.normalize(narr_rgb/255.))
            narr_seg_output = self.seg_head_narr(narr_embed)
            embed = torch.cat([
                wide_embed.mean(dim=[2,3]),
                self.bottleneck_narr(narr_embed.mean(dim=[2,3])),
            ], dim=1)
        else:
            embed = wide_embed.mean(dim=[2,3])


        # Action logits
        if self.all_speeds:
            act_output = self.act_head(embed).view(-1,self.num_cmds,self.num_speeds,self.num_steers+self.num_throts+1)
            act_output = action_logits(act_output, self.num_steers, self.num_throts)
        else:
            act_output = self.act_head(torch.cat([embed, self.spd_encoder(spd[:,None])], dim=1)).view(-1,self.num_cmds,1,self.num_steers+self.num_throts+1)
            act_output = action_logits(act_output, self.num_steers, self.num_throts).squeeze(2)

        if self.two_cam:
            return act_output, wide_seg_output, narr_seg_output
        else:
            return act_output, wide_seg_output


    @torch.no_grad()
    def policy(self, wide_rgb, narr_rgb, cmd, spd=None):
        
        assert (self.all_speeds and spd is None) or \
               (not self.all_speeds and spd is not None)
        
        wide_embed = self.backbone_wide(self.normalize(wide_rgb/255.))
        if self.two_cam:
            narr_embed = self.backbone_narr(self.normalize(narr_rgb/255.))
            embed = torch.cat([
                wide_embed.mean(dim=[2,3]),
                self.bottleneck_narr(narr_embed.mean(dim=[2,3])),
            ], dim=1)
        else:
            embed = wide_embed.mean(dim=[2,3])
        
        # Action logits
        if self.all_speeds:
            act_output = self.act_head(embed).view(-1,self.num_cmds,self.num_speeds,self.num_steers+self.num_throts+1)
            act_output = action_logits(act_output, self.num_steers, self.num_throts)
            
            # Action logits
            steer_logits = act_output[0,cmd,:,:self.num_steers]
            throt_logits = act_output[0,cmd,:,self.num_steers:self.num_steers+self.num_throts]
            brake_logits = act_output[0,cmd,:,-1]
        else:
            act_output = self.act_head(torch.cat([embed, self.spd_encoder(spd[:,None])], dim=1)).view(-1,self.num_cmds,1,self.num_steers+self.num_throts+1)
            act_output = action_logits(act_output, self.num_steers, self.num_throts).squeeze(2)
            
            # Action logits
            steer_logits = act_output[0,cmd,:self.num_steers]
            throt_logits = act_output[0,cmd,self.num_steers:self.num_steers+self.num_throts]
            brake_logits = act_output[0,cmd,-1]

        return steer_logits, throt_logits, brake_logits


def action_logits(raw_logits, num_steers, num_throts):
    
    steer_logits = raw_logits[...,:num_steers]
    throt_logits = raw_logits[...,num_steers:num_steers+num_throts]
    brake_logits = raw_logits[...,-1:]
    
    steer_logits = steer_logits.repeat(1,1,1,num_throts)
    throt_logits = throt_logits.repeat_interleave(num_steers,-1)
    
    act_logits = torch.cat([steer_logits + throt_logits, brake_logits], dim=-1)
    
    return act_logits



# if __name__ == '__main__':
    
#     from collections import namedtuple
#     Config = namedtuple('Config', [
#         'num_steers', 'num_throts', 'num_speeds', 'seg_channels', 'imagenet_pretrained'
#     ])
    
#     config = Config(num_steers=9, num_throts=3, num_speeds=4, seg_channels=[4,6,7,8,10], imagenet_pretrained=True)
#     model = TwoCameraModel(config)
    
#     wide_rgb = torch.zeros((1,3,128,480))
#     narr_rgb = torch.zeros((1,3,64,384))
    
#     act_output, seg_output = model(wide_rgb, narr_rgb)
#     print (act_output.shape, seg_output.shape)

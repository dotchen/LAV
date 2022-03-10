import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common.resnet import resnet18
from common.swin_transformer import ClassifySwin
from common.spatial_softmax import SpatialSoftmax
from copy import deepcopy

class UniPlanner(nn.Module):
    def __init__(self, bev_planner,
        pixels_per_meter=2, crop_size=64, x_offset=0, y_offset=0.75,
        feature_x_jitter=1, feature_angle_jitter=10, 
        num_plan=10, k=16, num_input_feature=96, num_out_feature=64, num_cmds=6, max_num_cars=4,
        num_plan_iter=1,
        ):

        super().__init__()

        self.num_cmds = num_cmds
        self.num_plan = num_plan
        self.num_plan_iter = num_plan_iter
        self.max_num_cars = max_num_cars

        self.bev_planner = bev_planner

        self.num_out_feature = num_out_feature

        self.pixels_per_meter = pixels_per_meter
        self.crop_size = crop_size

        self.feature_x_jitter = feature_x_jitter
        self.feature_angle_jitter = np.deg2rad(feature_angle_jitter)

        self.offset_x = nn.Parameter(torch.tensor(x_offset).float(), requires_grad=False)
        self.offset_y = nn.Parameter(torch.tensor(y_offset).float(), requires_grad=False)

        self.lidar_conv_emb = nn.Sequential(
            resnet18(num_channels=num_input_feature),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )

        self.plan_gru = nn.GRU(4,512,batch_first=True)
        self.plan_mlp = nn.Linear(512,2)

        self.cast_grus_ego = nn.ModuleList([nn.GRU(512, 64, batch_first=True) for _ in range(self.num_cmds)])
        self.cast_mlps_ego = nn.ModuleList([nn.Linear(64, 2) for _ in range(self.num_cmds)])

        self.cast_grus_other = nn.ModuleList([nn.GRU(512, 64, batch_first=True) for _ in range(self.num_cmds)])
        self.cast_mlps_other = nn.ModuleList([nn.Linear(64, 2) for _ in range(self.num_cmds)])
        self.cast_cmd_pred = nn.Sequential(
            nn.Linear(512,self.num_cmds),
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def ego_infer(self, features, pixels_per_meter=4, num_sample=50):
        cropped_ego_features = self.crop_feature(features[None], torch.zeros((1,2),dtype=features.dtype,device=features.device), torch.zeros((1,),dtype=features.dtype,device=features.device))
        pred_ego_locs = self.predict(cropped_ego_features, num_sample=num_sample)

        return pred_ego_locs[:,0]

    @torch.no_grad()
    def infer(self, features, det, cmd, nxp):
        """
        B (batch-size) is 1
        Note: This pixels_per_meter is on original scale
        self.pixels_per_meter is on feature map's scale
        """

        H = features.size(1)*2
        W = features.size(2)*2

        center_x = float(W/2 + self.offset_x*W/2)
        center_y = float(H/2 + self.offset_y*H/2)

        # Construct locs and oris
        locs, oris = [], []
        for X, Y, h, w, cos, sin in det:

            if np.linalg.norm([X-center_x,Y-center_y]) <= 4:
                continue

            # TODO: convert to ego's meters scale
            x = (X - center_x) / self.pixels_per_meter
            y = (Y - center_y) / self.pixels_per_meter
            o = float(np.arctan2(sin, cos))

            locs.append([x,y])
            oris.append(o)

        # relative locations and orientations
        locs = torch.tensor(locs, dtype=torch.float32).to(features.device)
        oris = torch.tensor(oris, dtype=torch.float32).to(features.device)

        N = len(locs)
        N_features = features.expand(N, *features.size())

        if N > 0:
            cropped_other_features = self.crop_feature(
                N_features, locs, oris, 
                pixels_per_meter=self.pixels_per_meter/2, 
                crop_size=self.crop_size
            )
            other_embd = self.lidar_conv_emb(cropped_other_features)
            other_cast_locs = self.cast(other_embd, mode='other')
            other_cast_cmds = self.cast_cmd_pred(other_embd)
            other_cast_locs = transform_points(other_cast_locs, oris[:,None].repeat(1,self.num_cmds))
            other_cast_locs += locs.view(N,1,1,2)
        else:
            other_cast_locs = torch.zeros((N,self.num_cmds,self.num_plan,2))
            other_cast_cmds = torch.zeros((N,self.num_cmds))

        cropped_ego_features = self.crop_feature(
            features[None], 
            torch.zeros((1,2),dtype=features.dtype,device=features.device), 
            torch.zeros((1,),dtype=features.dtype,device=features.device),
            pixels_per_meter=self.pixels_per_meter/2, crop_size=self.crop_size
        )
        ego_embd = self.lidar_conv_emb(cropped_ego_features)
        ego_cast_locs = self.cast(ego_embd, mode='ego')
        ego_plan_locs = self.plan(
            ego_embd, nxp[None], 
            cast_locs=ego_cast_locs,
            pixels_per_meter=self.pixels_per_meter, 
            crop_size=self.crop_size*2
        )[0,-1,cmd]

        return ego_plan_locs, ego_cast_locs[0,cmd], other_cast_locs, other_cast_cmds


    def _plan(self, embd, nxp, cast_locs, pixels_per_meter=4, crop_size=96):

        B = embd.size(0)

        h0, u0 = embd, nxp*pixels_per_meter/crop_size*2-1

        self.plan_gru.flatten_parameters()

        locs = []
        for i in range(self.num_cmds):
            u = torch.cat([
                u0.expand(self.num_plan, B, -1).permute(1,0,2),
                cast_locs[:,i]
            ], dim=2)
            
            out, _ = self.plan_gru(u, h0[None])
            locs.append(torch.cumsum(self.plan_mlp(out), dim=1))

        return torch.stack(locs, dim=1) + cast_locs
    
    def plan(self, embd, nxp, cast_locs=None, pixels_per_meter=4, crop_size=96):

        if cast_locs is None:
            plan_loc = self.cast(embd).detach()
        else:
            plan_loc = cast_locs.detach()
        
        plan_locs = []
        for i in range(self.num_plan_iter):
            plan_loc = self._plan(embd, nxp, plan_loc, pixels_per_meter=pixels_per_meter, crop_size=crop_size)
            plan_locs.append(plan_loc)

        return torch.stack(plan_locs, dim=1)

    def cast(self, embd, mode='ego'):
        B = embd.size(0)

        u = embd.expand(self.num_plan, B, -1).permute(1,0,2)

        if mode == 'ego':
            cast_grus = self.cast_grus_ego
            cast_mlps = self.cast_mlps_ego
        elif mode == 'other':
            # cast_grus = self.cast_grus_other
            # cast_mlps = self.cast_mlps_other
            cast_grus = self.cast_grus_ego
            cast_mlps = self.cast_mlps_ego

        locs = []
        for gru, mlp in zip(cast_grus, cast_mlps):
            gru.flatten_parameters()
            out, _ = gru(u)
            locs.append(torch.cumsum(mlp(out), dim=1))

        return torch.stack(locs, dim=1)

    def crop_feature(self, features, rel_locs, rel_oris, pixels_per_meter=4, crop_size=96):

        B, C, H, W = features.size()

        # ERROR proof hack...
        rel_locs = rel_locs.view(-1,2)

        rel_locs = rel_locs * pixels_per_meter/torch.tensor([H/2,W/2]).type_as(rel_locs).to(rel_locs.device)

        cos = torch.cos(rel_oris)
        sin = torch.sin(rel_oris)

        rel_x = rel_locs[...,0]
        rel_y = rel_locs[...,1]

        k = crop_size / H

        rot_x_offset = -k*self.offset_x*cos+k*self.offset_y*sin+self.offset_x
        rot_y_offset = -k*self.offset_x*sin-k*self.offset_y*cos+self.offset_y

        theta = torch.stack([
          torch.stack([k*cos, k*-sin, rot_x_offset+rel_x], dim=-1),
          torch.stack([k*sin, k*cos,  rot_y_offset+rel_y], dim=-1)
        ], dim=-2)
        

        grids = F.affine_grid(theta, torch.Size((B,C,crop_size,crop_size)), align_corners=True)

        cropped_features = F.grid_sample(features, grids, align_corners=True)

        return cropped_features

    def _make_downsample(self, num_in, num_out, stride=2):
        return nn.Sequential(
            nn.Conv2d(num_in,num_out,1,stride=stride),
            nn.BatchNorm2d(num_out),
        )

def transform_points(locs, oris):
    cos, sin = torch.cos(oris), torch.sin(oris)
    R = torch.stack([
        torch.stack([ cos, sin], dim=-1),
        torch.stack([-sin, cos], dim=-1),
    ], dim=-2)

    return locs @ R


def filter_cars(ego_locs, locs, typs):
    # We don't care about cars behind us ;)
    rel_locs = locs[:,:,0] - ego_locs[:,0:1]

    return typs & (rel_locs[...,1] < 0)


def random_sample(binaries, size):

    cut_binaries = torch.zeros_like(binaries)
    for i in range(binaries.size(0)):
        if binaries[i].sum() <= size:
            cut_binaries[i] = binaries[i]
        else:
            nonzero = torch.nonzero(binaries[i]).squeeze(1)
            nonzero_idx = torch.multinomial(torch.ones_like(nonzero).float(), size)
            nonzero = nonzero[nonzero_idx]
            cut_binaries[i,nonzero] = binaries[i,nonzero]
    
    return cut_binaries

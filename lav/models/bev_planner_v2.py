import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .resnet import resnet18

class BEVPlanner(nn.Module):
    def __init__(self,
        pixels_per_meter=2, crop_size=64, x_offset=0, y_offset=0.75,
        feature_x_jitter=1, feature_angle_jitter=10, 
        num_plan=10, k=16, num_out_feature=64, num_cmds=6, max_num_cars=5,
        num_plan_iter=1, num_frame_stack=0,
        ):

        super().__init__()
        
        self.num_cmds = num_cmds
        self.num_plan = num_plan
        self.num_plan_iter = num_plan_iter
        self.max_num_cars = max_num_cars

        self.num_out_feature = num_out_feature

        self.pixels_per_meter = pixels_per_meter
        self.crop_size = crop_size

        self.feature_x_jitter = feature_x_jitter
        self.feature_angle_jitter = np.deg2rad(feature_angle_jitter)

        self.offset_x = nn.Parameter(torch.tensor(x_offset).float(), requires_grad=False)
        self.offset_y = nn.Parameter(torch.tensor(y_offset).float(), requires_grad=False)

        self.bev_conv_emb = nn.Sequential(
            resnet18(num_channels=3+2*(num_frame_stack+1)),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )

        self.plan_gru = nn.GRU(4,512,batch_first=True)
        self.plan_mlp = nn.Linear(512,2)

        self.cast_grus = nn.ModuleList([nn.GRU(512, 64, batch_first=True) for _ in range(self.num_cmds)])
        self.cast_mlps = nn.ModuleList([nn.Linear(64, 2) for _ in range(self.num_cmds)])
        self.cast_cmd_pred = nn.Sequential(
            nn.Linear(512,self.num_cmds),
            nn.Sigmoid(),
        )

    def infer(self, bev, nxps):
        
        cropped_ego_bev = self.crop_feature(
            bev, 
            torch.zeros((1,2), dtype=bev.dtype,device=bev.device), 
            torch.zeros((1,),dtype=bev.dtype,device=bev.device),
            pixels_per_meter=self.pixels_per_meter, 
            crop_size=self.crop_size*2
        )

        ego_bev_embd = self.bev_conv_emb(cropped_ego_bev)

        ego_cast_locs = self.cast(ego_bev_embd)
        ego_plan_locs = self.plan(
            ego_bev_embd, nxps, 
            cast_locs=ego_cast_locs,
            pixels_per_meter=self.pixels_per_meter, 
            crop_size=self.crop_size*2
        )
        ego_cast_cmds = self.cast_cmd_pred(ego_bev_embd)
        
        return ego_plan_locs, ego_cast_locs, ego_cast_cmds
        
    def forward(self, bev, ego_locs, locs, oris, nxps, typs):

        ego_oris = oris[:,:1]

        locs = locs[:,1:]
        oris = oris[:,1:]
        typs = (typs[:,1:]==1) # 1 is for vehicles

        N = locs.size(1)

        # Only pick the good ones.
        typs = filter_cars(ego_locs, locs, typs)

        # Other vehicles
        if int(typs.float().sum()) > 0:

            # Guard against OOM: randomly sample cars to train on 
            typs = random_sample(typs, size=self.max_num_cars)

            # Flatten the locs
            flat_bev = bev.expand(N,*bev.size()).permute(1,0,2,3,4).contiguous()[typs]

            flat_locs = (locs[:,:,1:]-locs[:,:,:1])[typs]
            flat_rel_loc0 = (locs[:,:,0]-ego_locs[:,None,0])[typs]
            flat_rel_ori0 = (oris-ego_oris)[typs]

            K = flat_locs.size(0)

            locs_jitter = (torch.rand((K,2))*2-1).float().to(locs.device) * self.feature_x_jitter
            locs_jitter[:,1] = 0
            oris_jitter = (torch.rand((K,))*2-1).float().to(oris.device)  * self.feature_angle_jitter

            cropped_other_bev = self.crop_feature(flat_bev, flat_rel_loc0+locs_jitter, flat_rel_ori0+oris_jitter, pixels_per_meter=self.pixels_per_meter, crop_size=self.crop_size*2)

            other_locs = transform_points(flat_locs-locs_jitter[:,None], -flat_rel_ori0-oris_jitter)

            
            # import matplotlib.pyplot as plt
            # from matplotlib.pyplot import Circle
            # f, [ax1, ax2] = plt.subplots(1,2,figsize=(8,4))
            # ax1.imshow(bev[0].mean(0).detach().cpu().numpy())
            # ax2.imshow(cropped_other_bev[0].mean(0).detach().cpu().numpy())

            # for loc in other_locs[0]:
            #     ax2.add_patch(Circle(loc.detach().cpu().numpy()*4+[96,168],radius=2))

            # plt.show()

            other_bev_embd = self.bev_conv_emb(cropped_other_bev)

            other_cast_locs = self.cast(other_bev_embd)
            other_cast_cmds = self.cast_cmd_pred(other_bev_embd)

        else:
            dtype = bev.dtype
            device = bev.device

            other_locs = torch.zeros((N,self.num_plan,2), dtype=dtype, device=device)

            other_embd = torch.zeros((N,self.num_out_feature), dtype=dtype, device=device)
            other_bev_embd = torch.zeros((N,self.num_out_feature), dtype=dtype, device=device)

            other_cast_locs = torch.zeros((N,self.num_cmds,self.num_plan,2), dtype=dtype, device=device)
            other_cast_cmds = torch.zeros((N,self.num_cmds), dtype=dtype, device=device)

        B = bev.size(0)
        # locs_jitter = (torch.rand((B,2))*2-1).float().to(locs.device) * (0 if is_eval else self.feature_x_jitter)
        # locs_jitter[:,1] = 0
        # oris_jitter = (torch.rand((B,))*2-1).float().to(oris.device)  * (0 if is_eval else self.feature_angle_jitter)

        # ego_locs = transform_points(ego_locs[:,1:]-locs_jitter[:,None], -oris_jitter)
        # nxps     = transform_points(nxps[:,None]-locs_jitter[:,None], -oris_jitter)[:,0]

        # cropped_ego_bev = self.crop_feature(bev, locs_jitter, oris_jitter, pixels_per_meter=self.pixels_per_meter, crop_size=self.crop_size*2)
        cropped_ego_bev = self.crop_feature(
            bev, 
            torch.zeros((B,2), dtype=bev.dtype,device=bev.device), 
            torch.zeros((B,),dtype=bev.dtype,device=bev.device),
            pixels_per_meter=self.pixels_per_meter, 
            crop_size=self.crop_size*2
        )

        # f, [ax1, ax2] = plt.subplots(1,2,figsize=(8,4))
        # ax1.imshow(bev[0].mean(0).detach().cpu().numpy())
        # ax2.imshow(cropped_ego_bev[0].mean(0).detach().cpu().numpy())
        
        # plt.show()

        ego_bev_embd = self.bev_conv_emb(cropped_ego_bev)

        ego_cast_locs = self.cast(ego_bev_embd)
        ego_plan_locs = self.plan(
            ego_bev_embd, nxps, 
            cast_locs=ego_cast_locs,
            pixels_per_meter=self.pixels_per_meter, 
            crop_size=self.crop_size*2
        )
        ego_cast_cmds = self.cast_cmd_pred(ego_bev_embd)

        return (
            other_locs, other_cast_locs, other_cast_cmds,
            ego_plan_locs, ego_cast_locs, ego_cast_cmds
        )

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

    def cast(self, embd):
        B = embd.size(0)

        u = embd.expand(self.num_plan, B, -1).permute(1,0,2)

        locs = []
        for gru, mlp in zip(self.cast_grus, self.cast_mlps):
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

        # DEBUG
        # cos = torch.ones_like(cos)
        # sin = torch.zeros_like(sin)
        # END DEBUG
        
        # offset_x = self.offset_x + 

        rot_x_offset = -k*self.offset_x*cos+k*self.offset_y*sin+self.offset_x
        rot_y_offset = -k*self.offset_x*sin-k*self.offset_y*cos+self.offset_y

        # rel_x_offset = -k*self.offset_x*cos+k*self.offset_y*sin+self.offset_x
        # rel_y_offset = -k*self.offset_x*sin-k*self.offset_y*cos+self.offset_y
        # print (rel_x, rel_y)

        theta = torch.stack([
          torch.stack([k*cos, k*-sin, rot_x_offset+rel_x], dim=-1),
          torch.stack([k*sin, k*cos,  rot_y_offset+rel_y], dim=-1)
        ], dim=-2)
        

        grids = F.affine_grid(theta, torch.Size((B,C,crop_size,crop_size)), align_corners=True)

        # TODO: scale the grids??
        cropped_features = F.grid_sample(features, grids, align_corners=True)

        return cropped_features
    
        

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

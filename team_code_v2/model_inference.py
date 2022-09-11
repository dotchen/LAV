import numpy as np
import torch
import torch.nn as nn
import carla

from torch.nn import functional as F

from models.lidar import LiDARModel
from models.uniplanner import UniPlanner
from models.bev_planner import BEVPlanner

CAMERA_YAWS = [-60,0,60]

class InferModel(nn.Module):
    def __init__(self, lidar_model, uniplanner, camera_x, camera_z, device=torch.device("cuda")):
        super().__init__()

        self.uniplanner = uniplanner

        self.coord_converters = [torch.jit.script(CoordConverter(
            cam_yaw, lidar_xyz=[0,0,camera_z], cam_xyz=[camera_x,0,camera_z],
            rgb_h=288, rgb_w=256, fov=64
        ).to(device)) for cam_yaw in CAMERA_YAWS]

        self.lidar_model_point_pillar = lidar_model.point_pillar_net
        self.lidar_mode_backbone = torch.jit.script(lidar_model.backbone)
        self.lidar_center_head = torch.jit.script(lidar_model.center_head)
        self.lidar_box_head = torch.jit.script(lidar_model.box_head)
        self.lidar_ori_head = torch.jit.script(lidar_model.ori_head)
        self.lidar_seg_head = torch.jit.script(lidar_model.seg_head)

        self.lidar_conv_emb = torch.jit.script(uniplanner.lidar_conv_emb)
        self.plan = self.uniplanner.plan
        self.cast = self.uniplanner.cast
        self.cast_cmd_pred = self.uniplanner.cast_cmd_pred

        self.pixels_per_meter = self.uniplanner.pixels_per_meter
        self.offset_x = self.uniplanner.offset_x
        self.offset_y = self.uniplanner.offset_y
        self.crop_size = self.uniplanner.crop_size
        self.num_cmds = self.uniplanner.num_cmds
        self.num_plan = self.uniplanner.num_plan

    def forward_paint(self, cur_lidar, pred_sem):
        pred_sem = pred_sem[:,1:] * (1-pred_sem[:,:1])
        painted_lidar = self.point_painting(cur_lidar, pred_sem)

        fused_lidar = torch.cat([cur_lidar, painted_lidar], dim=-1)

        return fused_lidar


    def forward(self, lidar_points, nxps, cmd_value):

        features = self.lidar_model_point_pillar([lidar_points], [len(lidar_points)])
        features = self.lidar_mode_backbone(features)

        pred_heatmaps = self.lidar_center_head(features)
        pred_sizemaps = self.lidar_box_head(features)
        pred_orimaps = self.lidar_ori_head(features)
        pred_bev = self.lidar_seg_head(features)

        # Object detection
        det = self.det_inference(torch.sigmoid(pred_heatmaps[0]), pred_sizemaps[0], pred_orimaps[0])

        # Motion forecast & planning
        ego_embd, ego_plan_locs, ego_cast_locs, other_cast_locs, other_cast_cmds = self.uniplanner_infer(features[0], det[1], cmd_value, nxps)
        # ego_plan_locs = to_numpy(ego_plan_locs)
        # ego_cast_locs = to_numpy(ego_cast_locs)
        # other_cast_locs = to_numpy(other_cast_locs)
        # other_cast_cmds = to_numpy(other_cast_cmds)
        
        return ego_embd, ego_plan_locs, ego_cast_locs, other_cast_locs, other_cast_cmds, pred_bev, det

    def point_painting(self, lidar, sems):
    
        _, lidar_d = lidar.shape
        sem_c, sem_h, sem_w = sems[0].shape
    
        lidar_painted = torch.zeros((len(lidar), sem_c), dtype=torch.float, device=lidar.device)
    
        for sem, coord_converter in zip(sems, self.coord_converters):
    
            lidar_cam = coord_converter(lidar)
            lidar_cam_u, lidar_cam_v, lidar_cam_z = map(lambda x: x[...,0], torch.chunk(lidar_cam, 3, dim=-1))
            valid_idx = (lidar_cam_z>=0)&(lidar_cam_u>=0)&(lidar_cam_u<sem_w)&(lidar_cam_v>=0)&(lidar_cam_v<sem_h)
            lidar_cam = lidar_cam[valid_idx]
    
            lidar_sem = sem[:,lidar_cam[...,1],lidar_cam[...,0]].T

            lidar_painted[valid_idx] = lidar_sem
        
        return lidar_painted

    def det_inference(self, heatmaps, sizemaps, orimaps, min_score=0.2):

        dets = []
        for i, c in enumerate(heatmaps):
            det = []
            
            score, loc = extract_peak(c)
            peaks = [(float(s), int(l) % c.size(1), int(l) // c.size(1))
                for s, l in zip(score.cpu(), loc.cpu()) if s > min_score
            ]
            
            for s, x, y in peaks:
                w, h = float(sizemaps[0,y,x]),float(sizemaps[1,y,x])
                cos, sin = float(orimaps[0,y,x]), float(orimaps[1,y,x])
                
                if i==1 and max(w,h) < 0.1*self.pixels_per_meter:
                    continue
                
                # TODO: remove hardcode
                dist = np.linalg.norm([x-160,y-280])
                if dist <= 2 or dist >= 30*self.pixels_per_meter:
                    continue

                det.append((x,y,w,h,cos,sin))
            dets.append(det)
        
        return dets

    def uniplanner_infer(self, features, det, cmd_value, nxp):

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
            cropped_other_features = crop_feature(
                N_features, locs, oris, 
                pixels_per_meter=self.pixels_per_meter/2, 
                crop_size=self.crop_size,
                offset_x=self.offset_x,
                offset_y=self.offset_y,
            )
            other_embd = self.lidar_conv_emb(cropped_other_features)
            other_cast_locs = self.cast(other_embd, mode='other')
            other_cast_cmds = self.cast_cmd_pred(other_embd)
            other_cast_locs = transform_points(other_cast_locs, oris[:,None].repeat(1,self.num_cmds))
            other_cast_locs += locs.view(N,1,1,2)
        else:
            other_cast_locs = torch.zeros((N,self.num_cmds,self.num_plan,2))
            other_cast_cmds = torch.zeros((N,self.num_cmds))
    
        cropped_ego_features = crop_feature(
            features[None], 
            torch.zeros((1,2),dtype=features.dtype,device=features.device), 
            torch.zeros((1,),dtype=features.dtype,device=features.device),
            pixels_per_meter=self.pixels_per_meter/2, crop_size=self.crop_size,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
        )
        ego_embd = self.lidar_conv_emb(cropped_ego_features)
        ego_cast_locs = self.cast(ego_embd, mode='ego')
        ego_plan_locs = self.plan(
            ego_embd, nxp[None], 
            cast_locs=ego_cast_locs,
            pixels_per_meter=self.pixels_per_meter, 
            crop_size=self.crop_size*2
        )[0,-1,cmd_value]
    
        return ego_embd, ego_plan_locs, ego_cast_locs[0,cmd_value], other_cast_locs, other_cast_cmds

@torch.jit.script
def extract_peak(heatmap):

    # type: (Tensor,) -> Tuple[Tensor, Tensor]
    max_pool_ks=7
    max_det=15

    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks//2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)

    return score, loc

@torch.jit.script
def crop_feature(features, rel_locs, rel_oris, pixels_per_meter, crop_size, offset_x, offset_y):

    # type: (Tensor, Tensor, Tensor, float, int, float, float) -> Tensor

    B, C, H, W = features.size()

    # ERROR proof hack...
    rel_locs = rel_locs.view(-1,2)

    rel_locs = rel_locs * pixels_per_meter/torch.tensor([H/2,W/2]).type_as(rel_locs).to(rel_locs.device)

    cos = torch.cos(rel_oris)
    sin = torch.sin(rel_oris)

    rel_x = rel_locs[...,0]
    rel_y = rel_locs[...,1]

    k = crop_size / H

    rot_x_offset = -k*offset_x*cos+k*offset_y*sin+offset_x
    rot_y_offset = -k*offset_x*sin-k*offset_y*cos+offset_y

    theta = torch.stack([
      torch.stack([k*cos, k*-sin, rot_x_offset+rel_x], dim=-1),
      torch.stack([k*sin, k*cos,  rot_y_offset+rel_y], dim=-1)
    ], dim=-2)
    

    grids = F.affine_grid(theta, torch.Size((B,C,crop_size,crop_size)), align_corners=True)

    # TODO: scale the grids??
    cropped_features = F.grid_sample(features, grids, align_corners=True)

    return cropped_features

@torch.jit.script
def transform_points(locs, oris):

    # type: (Tensor, Tensor) -> Tensor

    cos, sin = torch.cos(oris), torch.sin(oris)
    R = torch.stack([
        torch.stack([ cos, sin], dim=-1),
        torch.stack([-sin, cos], dim=-1),
    ], dim=-2)

    return locs @ R



class CoordConverter(nn.Module):
    def __init__(self, cam_yaw, lidar_xyz=[0,0,2.5], cam_xyz=[1.4,0,2.5], rgb_h=320, rgb_w=320, fov=60):

        super().__init__()

        focal = rgb_w / (2.0 * np.tan(fov * np.pi / 360.0))
        
        K = torch.eye(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = rgb_w / 2.0
        K[1, 2] = rgb_h / 2.0

        lidar_to_world = torch.from_numpy(np.array(carla.Transform(
            carla.Location(*lidar_xyz),
        ).get_matrix())).float()

        world_to_cam = torch.from_numpy(np.array(carla.Transform(
            carla.Location(*cam_xyz),
            carla.Rotation(yaw=cam_yaw),
        ).get_inverse_matrix())).float()

        self.K = nn.Parameter(K)
        self.lidar_to_world = nn.Parameter(lidar_to_world)
        self.world_to_cam   = nn.Parameter(world_to_cam)

    def forward(self, lidar):

        # lidar_xyz = lidar[:,:3].T
        # lidar_xyz1 = np.r_[lidar_xyz, [np.ones(lidar_xyz.shape[1])]]
        lidar_xyz1 = torch.cat([lidar[:,:3], torch.ones_like(lidar[:,0:1])], dim=-1).T

        world = self.lidar_to_world @ lidar_xyz1
        cam   = self.world_to_cam @ world

        cam   = torch.stack([cam[1], -cam[2], cam[0]], dim=0)
        cam_2d = self.K @ cam

        cam_2d = torch.stack([
            cam_2d[0, :] / (1e-5+cam_2d[2, :]),
            cam_2d[1, :] / (1e-5+cam_2d[2, :]),
            cam_2d[2, :]], dim=0).T

        return cam_2d.long()


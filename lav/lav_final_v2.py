import yaml
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from einops import repeat

from lav.utils import _numpy
from lav.utils.point_painting import CoordConverter, point_painting

from lav.models.lidar import LiDARModel
from lav.models.loss import DetLoss
from lav.models.uniplanner import UniPlanner
from lav.models.bev_planner_v2 import BEVPlanner

PIXELS_PER_METER = 4

class LAV:
    def __init__(self, args):

        assert not (args.perceive_only and args.motion_only)

        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        # Save configs
        self.device = torch.device(args.device)

        self.lidar_model = LiDARModel(
            num_input=len(self.seg_channels)+self.num_frame_stack+10 if self.point_painting else self.num_frame_stack+10, 
            num_features=self.num_features,
            backbone=self.backbone,
            min_x=self.min_x, max_x=self.max_x,
            min_y=self.min_y, max_y=self.max_y,
            pixels_per_meter=self.pixels_per_meter,
        ).to(self.device)

        if not args.perceive_only:
            self.lidar_model.load_state_dict(torch.load(self.lidar_model_dir))

        bev_planner = BEVPlanner(
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size,
            feature_x_jitter=self.feature_x_jitter,
            feature_angle_jitter=self.feature_angle_jitter,
            x_offset=0, y_offset=1+self.min_x/((self.max_x-self.min_x)/2),
            num_cmds=self.num_cmds,
            num_plan=self.num_plan,
            num_plan_iter=self.num_plan_iter,
            num_frame_stack=self.num_frame_stack,
        ).to(self.device)
        bev_planner.load_state_dict(torch.load(self.bev_model_dir))
        bev_planner.eval()

        self.uniplanner = UniPlanner(
            bev_planner,
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size,
            feature_x_jitter=self.feature_x_jitter,
            feature_angle_jitter=self.feature_angle_jitter,
            x_offset=0, y_offset=1+self.min_x/((self.max_x-self.min_x)/2),
            num_cmds=self.num_cmds,
            num_plan=self.num_plan,
            num_input_feature=self.num_features[-1]*6,
            num_plan_iter=self.num_plan_iter,
        ).to(self.device)
        if not args.perceive_only and not args.motion_only:
            self.uniplanner.load_state_dict(torch.load(self.uniplanner_dir))

        params = list(self.uniplanner.plan_gru.parameters()) + \
                list(self.uniplanner.plan_mlp.parameters()) + \
                list(self.uniplanner.cast_grus_ego.parameters()) + \
                list(self.uniplanner.cast_mlps_ego.parameters()) + \
                list(self.uniplanner.cast_grus_other.parameters()) + \
                list(self.uniplanner.cast_mlps_other.parameters()) + \
                list(self.uniplanner.cast_cmd_pred.parameters()) + \
                list(self.uniplanner.lidar_conv_emb.parameters())

        if not args.motion_only:
            params += list(self.lidar_model.parameters())

        self.lidar_optim = optim.Adam(params, lr=args.lr)

        self.lidar_scheduler = StepLR(self.lidar_optim, step_size=4, gamma=0.5)

        # Send model to parallel
        if torch.cuda.device_count() > 1:
            self.lidar_model = nn.DataParallel(self.lidar_model)
            self.uniplanner = nn.DataParallel(self.uniplanner)
            self.multi_gpu = True
        else:
            self.multi_gpu = False

        # Misc
        W = (self.max_y-self.min_y)*self.pixels_per_meter
        H = (self.max_x-self.min_x)*self.pixels_per_meter
        self.bev_center = [
            W/2 + (self.min_y+self.max_y)/2*self.pixels_per_meter,
            H/2 + (self.min_x+self.max_x)/2*self.pixels_per_meter
        ]

        self.det_criterion = DetLoss()

        # TODO: Remove hard-coding
        self.seg_mask = self.build_seg_mask(
            h=H, w=W, cx=self.bev_center[0], cy=self.bev_center[1],
        ).to(self.device)

        self.branch_weights = torch.tensor(self.branch_weights).float().to(self.device)

        self.perceive_only = args.perceive_only
        self.motion_only = args.motion_only

        if not self.use_others_to_train:
            self.other_weight = 0.

    def state_dict(self, model_name):
        if model_name == 'bev':
            model = self.bev_planner
        elif model_name == 'seg':
            model = self.seg_model
        elif model_name == 'bra':
            model = self.bra_model
        elif model_name == 'lidar':
            model = self.lidar_model
        elif model_name == 'uniplanner':
            model = self.uniplanner
        else:
            raise NotImplementedError

        if self.multi_gpu:
            return model.module.state_dict()
        else:
            return model.state_dict()

    def train_lidar(self, 
        lidars, num_points,
        heatmaps, sizemaps, orimaps,
        bev,
        ego_locs, cmds, nxps, bras,
        locs, oris, typs, num_objs):

        lidars   = lidars.to(self.device)
        heatmaps = heatmaps.to(self.device)
        sizemaps = sizemaps.to(self.device)
        orimaps  = orimaps.to(self.device)

        ego_locs = ego_locs.float().to(self.device)
        cmds = cmds.long().to(self.device)
        nxps = nxps.float().to(self.device)
        locs = locs.float().to(self.device)
        oris = oris.float().to(self.device)
        typs = typs.to(self.device)

        bev = bev.float().to(self.device)
        seg_bev = bev[:,[0,1,2],...]

        cmds = cmds.long().to(self.device)
        idxs = (1-bras).bool().to(self.device)

        features,      \
        pred_heatmaps, \
        pred_sizemaps, \
        pred_orimaps,  \
        pred_bev = self.lidar_model(lidars, num_points)

        other_next_locs, other_cast_locs, other_cast_cmds, other_cast_locs_expert, other_cast_cmds_expert, \
        ego_next_locs, ego_plan_locs, ego_cast_locs, ego_cast_cmds, ego_cast_locs_expert, ego_plan_locs_expert = self.uniplanner(
            features, bev,
            ego_locs, locs, oris, nxps, typs,
        )

        # CenterNet/Point detection loss
        hm_loss, box_loss, ori_loss = self.det_criterion(
            pred_heatmaps, heatmaps,
            pred_sizemaps, sizemaps,
            pred_orimaps , orimaps,
        )

        # Detection loss
        det_loss = hm_loss + self.box_weight * box_loss + self.ori_weight * ori_loss

        # Segmentation loss
        seg_loss = torch.mean(F.binary_cross_entropy(pred_bev, seg_bev, reduction='none') * self.seg_mask) * self.seg_weight

        # Motion loss
        # special_cmds = (cmds!=3)
        # import pdb; pdb.set_trace()
        # plan_loss = torch.mean(F.l1_loss(ego_plan_locs, ego_next_locs[:,None,None].repeat(1,self.num_plan_iter,self.num_cmds,1,1), reduction='none').mean(dim=[1,2,3,4])[idxs] * self.branch_weights[cmds[idxs]])
                #   + F.l1_loss(ego_plan_locs, ego_plan_locs_expert[:,-1].gather(1,cmds.expand(self.num_plan,2,1,-1).permute(3,2,0,1)).unsqueeze(1).repeat(1,self.num_plan_iter,self.num_cmds,1,1))

        plan_loss = torch.mean(
            F.l1_loss(
                ego_plan_locs, 
                ego_plan_locs_expert[:,-1].gather(1,cmds.expand(self.num_plan,2,1,-1).permute(3,2,0,1)).unsqueeze(1).repeat(1,self.num_plan_iter,self.num_cmds,1,1), reduction='none'
            ).mean(dim=[1,2,3,4]) * self.branch_weights[cmds]
        )

        if self.distill:
            ego_cast_loss = F.l1_loss(ego_cast_locs, ego_cast_locs_expert)
            other_cast_loss = F.l1_loss(other_cast_locs, other_cast_locs_expert)
            cmd_loss = F.binary_cross_entropy(other_cast_cmds, other_cast_cmds_expert)
        else:
            ego_cast_loss = F.l1_loss(ego_cast_locs.gather(1,cmds.expand(self.num_plan,2,1,-1).permute(3,2,0,1)).squeeze(1), ego_locs[:,1:], reduction='none').mean(dim=[1,2])[idxs].mean()
            other_cast_losses = F.l1_loss(other_cast_locs, other_next_locs.unsqueeze(1).repeat(1,self.num_cmds,1,1), reduction='none').mean(dim=[2,3])
            other_cast_loss = other_cast_losses.min(1)[0].mean()
            cmd_loss = F.binary_cross_entropy(ego_cast_cmds, (1.-self.cmd_smooth) * F.one_hot(cmds, self.num_cmds) + self.cmd_smooth / self.num_cmds) 

        mot_loss = plan_loss + ego_cast_loss + other_cast_loss * self.other_weight + cmd_loss * self.cmd_weight

        if self.perceive_only:
            loss = det_loss + seg_loss
        elif self.motion_only:
            loss = mot_loss
        else:
            loss = mot_loss + (det_loss + seg_loss) * self.perception_weight

        self.lidar_optim.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.lidar_model.parameters(), 40.)
        self.lidar_optim.step()

        # inference
        det = self.det_inference(torch.sigmoid(pred_heatmaps[0]), pred_sizemaps[0], pred_orimaps[0], break_tie=True)
        gt_det = self.det_inference(heatmaps[0], sizemaps[0], orimaps[0])

        ego_plan_locs, other_cast_locs, other_cast_cmds = self.mot_inference(
            lidars[0], num_points[0], cmds[0], nxps[0],
            # heatmaps=heatmaps,  # Comment this to use predicted detections
            # sizemaps=sizemaps,  # Comment this to use predicted detections
            # orimaps=orimaps,    # Comment this to use predicted detections
        )

        return dict(
            hm_loss=float(hm_loss),
            box_loss=float(box_loss),
            ori_loss=float(ori_loss),
            seg_loss=float(seg_loss),
            plan_loss=float(plan_loss),
            ego_cast_loss=float(ego_cast_loss),
            other_cast_loss=float(other_cast_loss),
            cmd_loss=float(cmd_loss),
            pred_bev=_numpy(pred_bev[0]).mean(axis=0),
            bev=_numpy(bev[0].mean(axis=0)),
            det=det,
            gt_det=gt_det,
            ego_plan_locs=_numpy(ego_plan_locs)*self.pixels_per_meter + self.bev_center,
            ego_next_locs=_numpy(ego_locs[0])*self.pixels_per_meter + self.bev_center,
            other_next_locs=_numpy(locs[0])*self.pixels_per_meter + self.bev_center,
            other_cast_locs=_numpy(other_cast_locs)*self.pixels_per_meter + self.bev_center,
            other_cast_cmds=_numpy(other_cast_cmds),
            cmd=int(cmds[0]),
            nxp=_numpy(nxps[0])*self.pixels_per_meter + self.bev_center,
            num_points=_numpy(num_points[0]),
        )

    def build_seg_mask(self, w=320, h=320, cx=160, cy=280, radius_x=240, radius_y=240):

        x = torch.arange(w)
        y = torch.arange(h)

        gx = (-((x[:, None] - cx) / radius_x)**2).exp()
        gy = (-((y[:, None] - cy) / radius_y)**2).exp()

        gaussian, id = (gx[None] * gy[:, None]).max(dim=-1)

        return gaussian
    
    @torch.no_grad()
    def det_inference(self, heatmaps, sizemaps, orimaps, **kwargs):

        dets = []
        for i, c in enumerate(heatmaps):
            det = []
            for s, x, y in extract_peak(c, **kwargs):
                w, h = float(sizemaps[0,y,x]),float(sizemaps[1,y,x])
                cos, sin = float(orimaps[0,y,x]), float(orimaps[1,y,x])
                
                if i==1 and w < 0.1*self.pixels_per_meter or h < 0.2*self.pixels_per_meter:
                    continue

                det.append((x,y,w,h,cos,sin))
            dets.append(det)
        return dets
    
    @torch.no_grad()
    def mot_inference(self, lidar, num_point, cmd, nxp, heatmaps=None, sizemaps=None, orimaps=None):

        self.lidar_model.eval()
        self.uniplanner.eval()

        if self.multi_gpu:
            uniplanner = self.uniplanner.module
        else:
            uniplanner = self.uniplanner

        features,      \
        pred_heatmaps, \
        pred_sizemaps, \
        pred_orimaps,  \
        pred_bev = self.lidar_model(lidar[None], num_point[None])

        if heatmaps is None:
            heatmaps = torch.sigmoid(pred_heatmaps)

        if sizemaps is None:
            sizemaps = pred_sizemaps

        if orimaps is None:
            orimaps = pred_orimaps

        det = self.det_inference(heatmaps[0], sizemaps[0], orimaps[0])

        ego_plan_locs, _, other_cast_locs, other_cast_cmds = uniplanner.infer(features[0], det[1], cmd, nxp)

        self.lidar_model.train()
        self.uniplanner.train()

        return ego_plan_locs, other_cast_locs, other_cast_cmds



def extract_peak(heatmap, max_pool_ks=7, min_score=0.2, max_det=20, break_tie=False):
    
    if break_tie:
        heatmap = heatmap + 1e-7*torch.randn(*heatmap.size(), device=heatmap.device)
    
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks//2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)

    return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
            for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]

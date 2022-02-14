import yaml
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from einops import repeat
from lav.utils import _numpy
from lav.utils.point_painting import CoordConverter, point_painting

from lav.models.rgb import RGBSegmentationModel, RGBBrakePredictionModel
from lav.models.lidar import LiDARModel
from lav.models.bev_planner import BEVPlanner

PIXELS_PER_METER = 4

class LAV:
    def __init__(self, args):

        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        # Save configs
        self.device = torch.device(args.device)

        # Models
        self.seg_model = RGBSegmentationModel(self.seg_channels).to(self.device)
        self.bra_model = RGBBrakePredictionModel([4,10,18], pretrained=True).to(self.device)
        
        self.bev_planner = BEVPlanner(
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size,
            feature_x_jitter=self.feature_x_jitter,
            feature_angle_jitter=self.feature_angle_jitter,
            x_offset=0, y_offset=1+self.min_x/((self.max_x-self.min_x)/2),
            num_cmds=self.num_cmds,
            num_plan=self.num_plan,
            num_plan_iter=self.num_plan_iter,
        ).to(self.device)

        # Optimizers
        self.seg_optim = optim.Adam(self.seg_model.parameters(), lr=args.lr)
        self.bra_optim = optim.Adam(self.bra_model.parameters(), lr=args.lr)

        self.bev_optim = optim.Adam(self.bev_planner.parameters(), lr=args.lr)
        self.bev_scheduler = StepLR(self.bev_optim, step_size=32, gamma=0.5)

        # Send model to parallel
        if torch.cuda.device_count() > 1:
            self.seg_model = nn.DataParallel(self.seg_model)
            self.bra_model = nn.DataParallel(self.bra_model)
            self.bev_planner = nn.DataParallel(self.bev_planner)
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

    def load_bev(self):
        state_dict = torch.load(self.bev_model_dir)
        if self.multi_gpu:
            self.uniplanner.module.load_state_dict(state_dict)
            self.uniplanner.module.bev_conv_emb.eval()
        else:
            self.uniplanner.load_state_dict(state_dict)

    def train_bev(self, bev, ego_locs, cmds, nxps, bras, locs, oris, typs, num_objs):

        bev = bev.float().to(self.device)
        ego_locs = ego_locs.float().to(self.device)
        nxps = nxps.float().to(self.device)
        cmds = cmds.long().to(self.device)
        idxs = (1-bras).bool().to(self.device)

        locs = locs.float().to(self.device)
        oris = oris.float().to(self.device)
        typs = typs.to(self.device)

        other_next_locs, other_cast_locs, other_cast_cmds, \
        ego_plan_locs, ego_cast_locs, ego_cast_cmds = self.bev_planner(
            bev,
            ego_locs, locs, oris, nxps, typs,
        )

        special_cmds = (cmds!=3)
        plan_loss = F.l1_loss(ego_plan_locs, repeat(ego_locs[:,1:], "b t d -> b i c t d", i=self.num_plan_iter, c=self.num_cmds, d=2))

        ego_cast_loss = F.l1_loss(ego_cast_locs.gather(1,repeat(cmds, "b -> b 1 t d",  t=self.num_plan, d=2)).squeeze(1), ego_locs[:,1:])
        other_cast_losses = F.l1_loss(other_cast_locs, repeat(other_next_locs, "b t d -> b c t d", c=self.num_cmds), reduction="none").mean(dim=[2,3])
        other_cast_loss = other_cast_losses.min(1)[0].mean()

        cmd_loss = F.binary_cross_entropy(ego_cast_cmds, F.one_hot(cmds, self.num_cmds).float())

        loss = plan_loss + ego_cast_loss + other_cast_loss + cmd_loss * self.cmd_weight

        self.bev_optim.zero_grad()
        loss.backward()
        self.bev_optim.step()

        return dict(
            plan_loss=float(plan_loss),
            ego_cast_loss=float(ego_cast_loss),
            other_cast_loss=float(other_cast_loss),
            cmd_loss=float(cmd_loss),
            bev=_numpy(bev[0].mean(axis=0)),
            ego_plan_locs=_numpy(ego_plan_locs[0,-1,int(cmds[0])])*self.pixels_per_meter + self.bev_center,
            ego_cast_locs=_numpy(ego_cast_locs[0])*self.pixels_per_meter + self.bev_center,
            ego_cast_cmds=_numpy(ego_cast_cmds[0]),
            nxp=_numpy(nxps[0])*self.pixels_per_meter + self.bev_center,
            cmd=int(cmds[0]),
        )


    def train_seg(self, rgb, sem):

        rgb = rgb.float().permute(0,3,1,2).to(self.device)
        sem = sem.long().to(self.device)

        pred_sem = self.seg_model(rgb)

        loss = F.cross_entropy(pred_sem, sem)

        self.seg_optim.zero_grad()
        loss.backward()
        self.seg_optim.step()

        opt_info = dict(
            loss=float(loss),
            rgb=_numpy(rgb[0].permute(1,2,0).byte()),
            sem=_numpy(sem[0]),
            pred_sem=_numpy(pred_sem[0]).argmax(0),
        )

        del rgb, sem, pred_sem, loss

        return opt_info


    def train_bra(self, rgb1, rgb2, sem1, sem2, bra):

        rgb1 = rgb1.float().permute(0,3,1,2).to(self.device)
        sem1 = sem1.long().to(self.device)
        rgb2 = rgb2.float().permute(0,3,1,2).to(self.device)
        sem2 = sem2.long().to(self.device)
        bra = bra.float().to(self.device)

        pred_bra, pred_sem1, pred_sem2 = self.bra_model(rgb1, rgb2, mask=True)

        loss = F.binary_cross_entropy(pred_bra, bra) \
             + 1/2*F.cross_entropy(pred_sem1, sem1) \
             + 1/2*F.cross_entropy(pred_sem2, sem2)

        self.bra_optim.zero_grad()
        loss.backward()
        self.bra_optim.step()

        opt_info = dict(
            loss=float(loss),
            rgb1=_numpy(rgb1[0].permute(1,2,0).byte()),
            rgb2=_numpy(rgb2[0].permute(1,2,0).byte()),
            bra=float(bra[0]),
            pred_bra=float(pred_bra[0]),
            pred_sem1=_numpy(pred_sem1[0]).argmax(0),
            pred_sem2=_numpy(pred_sem2[0]).argmax(0),
        )

        del rgb1, sem1, rgb2, sem2, bra, loss

        return opt_info


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
    def mot_inference(self, pillar, num_point, coord, cmd, nxp, heatmaps=None, sizemaps=None, orimaps=None):

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
        pred_bev = self.lidar_model(pillar[None], num_point[None], coord[None])

        if heatmaps is None:
            heatmaps = torch.sigmoid(pred_heatmaps)

        if sizemaps is None:
            sizemaps = pred_sizemaps

        if orimaps is None:
            orimaps = pred_orimaps

        det = self.det_inference(heatmaps[0], sizemaps[0], orimaps[0])

        ego_plan_locs, other_cast_locs, other_cast_cmds = uniplanner.infer(features[0], det[1], cmd, nxp, pixels_per_meter=self.pixels_per_meter)

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

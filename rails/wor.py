import ray
import math
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch import nn, optim
from .models import EgoModel, CameraModelV2
from .bellman import BellmanUpdater
from .datasets.main_dataset import MainDataset
from .utils import to_numpy

class WOR:
    def __init__(self, args):

        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        for key, value in config.items():
            setattr(self, key, value)
        
        # Save configs
        self.device = torch.device(args.device)
        self.ego_traj_len = self.num_plan

        # Create models
        self.ego_model  = EgoModel(dt=1./args.fps*(args.num_repeat+1)).to(args.device)
        self.main_model = CameraModelV2(config).to(args.device)

        if args.device=='cuda' and torch.cuda.device_count() > 1:
            self.main_model = nn.DataParallel(self.main_model)
            self.multi_gpu = True
        else:
            self.multi_gpu = False
        
        # Create optimizers
        self.ego_optim  = optim.Adam(self.ego_model.parameters(), lr=args.lr)
        self.main_optim = optim.Adam(self.main_model.parameters(), lr=args.lr)

        BellmanUpdater.setup(config, self.ego_model, device=self.device)
        
    def main_model_state_dict(self):
        if self.multi_gpu:
            return self.main_model.module.state_dict()
        else:
            return self.main_model.state_dict()

        
    def train_main(self, wide_rgbs, wide_sems, narr_rgbs, narr_sems, act_vals, spds, cmds):

        wide_rgbs = wide_rgbs.float().permute(0,3,1,2).to(self.device)
        narr_rgbs = narr_rgbs.float().permute(0,3,1,2).to(self.device)
        wide_sems = wide_sems.long().to(self.device)
        narr_sems = narr_sems.long().to(self.device)
        act_vals  = act_vals.float().permute(0,1,3,2).to(self.device)
        spds      = spds.float().to(self.device)
        cmds      = cmds.long().to(self.device)

        act_probs = F.softmax(act_vals/self.temperature, dim=3)
        
        if self.use_narr_cam:
            act_outputs, wide_seg_outputs, narr_seg_outputs = self.main_model(wide_rgbs, narr_rgbs, spd=None if self.all_speeds else spds)
        else:
            act_outputs, wide_seg_outputs = self.main_model(wide_rgbs, narr_rgbs, spd=None if self.all_speeds else spds)
        
        if self.all_speeds:
            act_loss = F.kl_div(F.log_softmax(act_outputs, dim=3), act_probs, reduction='none').mean(dim=[2,3])
        else:
            act_probs = self.spd_lerp(act_probs, spds)
            act_loss = F.kl_div(F.log_softmax(act_outputs, dim=2), act_probs, reduction='none').mean(dim=2)
        
        turn_loss = (act_loss[:,0]+act_loss[:,1]+act_loss[:,2]+act_loss[:,3])/4
        lane_loss = (act_loss[:,4]+act_loss[:,5]+act_loss[:,3])/3
        foll_loss = act_loss[:,3]
        
        is_turn = (cmds==0)|(cmds==1)|(cmds==2)
        is_lane = (cmds==4)|(cmds==5)

        act_loss = torch.mean(torch.where(is_turn, turn_loss, foll_loss) + torch.where(is_lane, lane_loss, foll_loss))
        seg_loss = F.cross_entropy(wide_seg_outputs, wide_sems)

        if self.use_narr_cam:
            seg_loss = seg_loss + F.cross_entropy(narr_seg_outputs, narr_sems)
            seg_loss = seg_loss / 2

        loss = act_loss + self.seg_weight * seg_loss
        
        # Backpropogate
        self.main_optim.zero_grad()
        loss.backward()
        self.main_optim.step()
        
        if self.all_speeds:
            act_prob      = BellmanUpdater._batch_lerp(act_probs[0,int(cmds[0])].permute(1,0), spds[0:1], min_val=BellmanUpdater._min_speeds, max_val=BellmanUpdater._max_speeds)
            pred_act_prob = BellmanUpdater._batch_lerp(F.softmax(act_outputs[0,int(cmds[0])],dim=1).permute(1,0), spds[0:1], min_val=BellmanUpdater._min_speeds, max_val=BellmanUpdater._max_speeds)
        else:
            act_prob      = act_probs[0,int(cmds[0])]
            pred_act_prob = F.softmax(act_outputs[0,int(cmds[0])], dim=0)
        
        return dict(
            act_loss=float(act_loss),
            seg_loss=float(seg_loss),
            gt_seg=to_numpy(wide_sems[0]),
            pred_seg  =to_numpy(wide_seg_outputs[0]).argmax(0),
            cmd     =int(cmds[0]),
            spd     =float(spds[0]),
            wide_rgb=to_numpy(wide_rgbs[0].permute(1,2,0).byte()),
            narr_rgb=to_numpy(narr_rgbs[0].permute(1,2,0).byte()),
            act_prob=to_numpy(act_prob[:-1]).reshape(self.num_throts,self.num_steers),
            pred_act_prob=to_numpy(pred_act_prob[:-1]).reshape(self.num_throts,self.num_steers),
            act_brak=float(act_prob[-1]),
            pred_act_brak=float(pred_act_prob[-1])
        )


    def train_ego(self, locs, rots, spds, acts):
        
        locs = locs[...,:2].to(self.device)
        yaws = rots[...,2:].to(self.device) * math.pi / 180.
        spds = spds.to(self.device)
        acts = acts.to(self.device)
        
        pred_locs = []
        pred_yaws = []
        
        pred_loc = locs[:,0]
        pred_yaw = yaws[:,0]
        pred_spd = spds[:,0]
        for t in range(locs.shape[1]-1):
            act = acts[:,t]
            
            pred_loc, pred_yaw, pred_spd = self.ego_model(pred_loc, pred_yaw, pred_spd, act)
            
            pred_locs.append(pred_loc)
            pred_yaws.append(pred_yaw)
        
        pred_locs = torch.stack(pred_locs, 1)
        pred_yaws = torch.stack(pred_yaws, 1)

        loc_loss = F.l1_loss(pred_locs, locs[:,1:])
        ori_loss = F.l1_loss(torch.cos(pred_yaws), torch.cos(yaws[:,1:])) + F.l1_loss(torch.sin(pred_yaws), torch.sin(yaws[:,1:]))
        
        loss = loc_loss + ori_loss
        
        # Backprop gradient
        self.ego_optim.zero_grad()
        loss.backward()
        self.ego_optim.step()
        
        return dict(
            loc_loss=float(loc_loss),
            ori_loss=float(ori_loss),
            pred_locs=to_numpy(pred_locs),
            pred_yaws=to_numpy(pred_yaws),
            locs=to_numpy(locs),
            yaws=to_numpy(yaws),
        )

    def bellman_plan(self, lbls, locs, rots, spds, cmd, visualize=None, dense_action_values=False):
        """
        Everything is numpy or (int, float)
        """

        yaw = rots[0]
        delta_locs, delta_yaws, next_spds = BellmanUpdater.compute_table(yaw/180*math.pi)

        cmds = range(6)

        loc_offsets = locs[:1,:2] - locs[1:,:2]
        waypoint_rews, stop_rews, brak_rews, frees = self.get_reward(lbls, loc_offsets, yaw/180*math.pi)
        
        _action_values = []
        action_values  = []
        for i in cmds:
            _action_value, action_value = BellmanUpdater.get_action(
                delta_locs, delta_yaws, next_spds,
                waypoint_rews[...,i], brak_rews, stop_rews, frees,
                torch.tensor(locs[1:,:2]-locs[:1,:2]).to(BellmanUpdater._device), 
                extract=(
                    torch.tensor([[0.,0.]]*len(self.camera_yaws)),  # location
                    torch.tensor([math.pi/180*yaw for yaw in self.camera_yaws]),       # yaw
                    torch.tensor([spds[0]]*len(self.camera_yaws)),  # spd
                ),
                visualize=visualize if i==3 else None,
                dense_action_values=dense_action_values
            )
            
            _action_values.append(to_numpy(_action_value))
            action_values.append(to_numpy(action_value))

        return _action_values[cmd], np.stack(action_values, axis=1)

    def get_reward(self, lbls, loc_offsets, ref_yaw):
        
        waypoint_rews = []
        stop_rews = []
        brak_rews = []
        frees = []
        
        for lbl, loc_offset in zip(lbls, loc_offsets):    
            waypoint_rew, stop_rew, brak_rew, free = BellmanUpdater.get_reward(lbl, loc_offset, ref_yaw=ref_yaw)

            waypoint_rews.append(waypoint_rew)
            stop_rews.append(stop_rew)
            brak_rews.append(brak_rew)
            frees.append(free)

        return torch.stack(waypoint_rews), torch.stack(stop_rews), torch.stack(brak_rews), torch.stack(frees)
    
    def spd_lerp(self, v, x):

        D = v.shape[2]

        min_val = self.min_speeds
        max_val = self.max_speeds
        
        x = (x - min_val)/(max_val - min_val)*(D-1)

        x0, x1 = x.floor().long(), x.ceil().long()
        x0 = torch.clamp(x0,min=0,max=D-1)
        x1 = torch.clamp(x1,min=0,max=D-1)

        w = x - x0

        x0 = x0.expand(v.shape[1],1,v.shape[3],-1).permute(3,0,1,2)
        x1 = x1.expand(v.shape[1],1,v.shape[3],-1).permute(3,0,1,2)

        return (1-w[:,None,None]) * v.gather(2,x0).squeeze(2) + w[:,None,None] * v.gather(2,x1).squeeze(2)

@ray.remote(num_gpus=1./2)
class WORActionLabeler():
    def __init__(self, args, dataset, worker_id=0, total_worker=4):

        assert 0 <= worker_id < total_worker
        
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        self._wor = WOR(args)
        self.write_dataset = dataset
        self.dataset = MainDataset(args.data_dir, args.config_path)

        # Allocate idxes
        total_frames = ray.get(dataset.num_frames.remote())
        self.start_idx = total_frames//total_worker*worker_id
        self.end_idx = min(total_frames//total_worker*(worker_id+1), total_frames)
        if worker_id == total_worker-1:
            self.end_idx = max(self.end_idx, total_frames)

        # print (self.start_idx, self.end_idx)
        # DEBUG
        # self.start_idx = 
        # self.end_idx = self.start_idx + 100
        # END DEBUG

        self.worker_id = worker_id
        self.num_per_log = args.num_per_log
        
        self.camera_yaws = self._wor.camera_yaws
        
    def run(self, logger):

        count = 0
        for idx in range(self.start_idx, self.end_idx):
            wide_rgb, wide_sem, narr_rgb, lbls, locs, rots, spds, cmd = self.dataset[idx*len(self.camera_yaws)]

            action_value, action_values = self._wor.bellman_plan(lbls, locs, rots, spds, cmd, dense_action_values=False)

            count += 1

            if count % self.num_per_log == 0:
                logger.log_label_info.remote(dict(
                    wide_rgb=wide_rgb,
                    narr_rgb=narr_rgb,
                    act_val_norm=action_value[0,:-1].reshape(self._wor.num_throts, self._wor.num_steers),
                    act_val_brak=float(action_value[0,-1]),
                    cmd=cmd,
                ), count, worker_id=self.worker_id)
            
            self.write_dataset.save.remote(idx*len(self.camera_yaws), action_values)
        
        logger.log_label_info.remote(dict(
            wide_rgb=wide_rgb,
            narr_rgb=narr_rgb,
            act_val_norm=action_value[0,:-1].reshape(self._wor.num_throts, self._wor.num_steers),
            act_val_brak=float(action_value[0,-1]),
            cmd=cmd,
        ), count, worker_id=self.worker_id)

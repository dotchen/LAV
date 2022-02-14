import torch
from torch import nn
from torch.nn import functional as F

class DetLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.hm_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.ori_criterion = nn.SmoothL1Loss(reduction='none')
        self.box_criterion = nn.SmoothL1Loss(reduction='none')

    def forward(self, 
            pred_heatmaps, heatmaps,
            pred_sizemaps, sizemaps,
            pred_orimaps , orimaps,
        ):

        size_w, _ = heatmaps.max(dim=1, keepdim=True)
        p_det = torch.sigmoid(pred_heatmaps * (1-2*heatmaps))
        
        det_loss = (self.hm_criterion(pred_heatmaps, heatmaps)*p_det).mean() / p_det.mean()
        box_loss = (size_w * self.box_criterion(pred_sizemaps, sizemaps)).mean() / size_w.mean()

        ori_loss = (size_w * self.ori_criterion(pred_orimaps, orimaps)).mean() / size_w.mean()

        return det_loss, box_loss, ori_loss


class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_bev, bev):

        return self.criterion(pred_bev, bev).mean()
        
class MotLoss(nn.Module):
    def __init__(self, distill, smooth):
        super().__init__()

        self.bc_criterion = nn.L1Loss(reduction='none')
        self.cmd_criterion = nn.BCELoss()
        self.distill = distill
        self.smooth = smooth

    def forward(self, plan_locs, cast_locs, locs, pred_cmds, expert_locs, expert_cmds, cmds, idxs=None):

        T = locs.size(1)
        N = pred_cmds.size(1)

        plan_locs = plan_locs.gather(1, cmds.expand(T,2,1,-1).permute(3,2,0,1)).squeeze(1)
        plan_losses = self.bc_criterion(plan_locs, locs).mean(dim=[1,2])

        if self.distill:
            cast_loss = self.bc_criterion(cast_locs, expert_locs.detach()).mean()
            cmd_loss = self.cmd_criterion(pred_cmds, expert_cmds.detach())
        else:
            cast_locs = cast_locs.gather(1, cmds.expand(T,2,1,-1).permute(3,2,0,1)).squeeze(1)
            cast_loss   = self.bc_criterion(cast_locs, locs).mean()

            cmds_label = (1.-self.smooth) * F.one_hot(cmds, N) + self.smooth / N
            cmd_loss = self.cmd_criterion(pred_cmds, cmds_label)

        if idxs is None:
            plan_loss = plan_losses.mean()
        else:
            plan_loss = plan_losses[idxs].mean()

        return (plan_loss + cast_loss) / 2, cmd_loss

    def others_forward(self, cast_locs, expert_locs, locs):

        if self.distill:
            return self.bc_criterion(cast_locs, expert_locs).mean()
        else:
            other_bc_losses = self.bc_criterion(cast_locs, locs).mean(dim=[2,3])
            return other_bc_losses.min(1)[0].mean()

    def bev_forward(self, plan_locs, cast_locs, locs, pred_cmds, cmds, idxs=None):

        T = locs.size(1)
        N = pred_cmds.size(1)

        plan_locs = plan_locs.gather(1, cmds.expand(T,2,1,-1).permute(3,2,0,1)).squeeze(1)
        plan_losses = self.bc_criterion(plan_locs, locs).mean(dim=[1,2])
        
        cast_locs = cast_locs.gather(1, cmds.expand(T,2,1,-1).permute(3,2,0,1)).squeeze(1)
        cast_loss   = self.bc_criterion(cast_locs, locs).mean()

        cmd_loss = self.cmd_criterion(pred_cmds, F.one_hot(cmds, N).float())

        if idxs is None:
            plan_loss = plan_losses.mean()
        else:
            plan_loss = plan_losses[idxs].mean()

        return (plan_loss + cast_loss) / 2, cmd_loss

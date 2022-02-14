import yaml
import wandb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arrow
from torch import nn

from .visualization import visualize_semantic, visualize_semantic_processed

PIXELS_AHEAD_VEHICLE = 120
ARROW_WIDTH = 10
# Tango colors
ORANGE = '#fcaf3e'
RED    = '#cc0000'
BLUE   = '#3465a4'
GREEN  = '#73d216'
BLACK  = '#000000'

class Logger:
    def __init__(self, wandb_project, args):

        # Read configs
        with open(args.config_path, 'rb') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        wandb.init(project=wandb_project, config=config)

    def watch_model(self, model):
        wandb.watch(model, log_freq=100)

    @property
    def save_dir(self):
        return wandb.run.dir

    def log_bra_info(self, it, opt_info):

        rgb1 = opt_info.pop('rgb1')
        rgb2 = opt_info.pop('rgb2')
        bra = opt_info.pop('bra')
        pred_bra = opt_info.pop('pred_bra')
        pred_sem1 = opt_info.pop('pred_sem1')
        pred_sem2 = opt_info.pop('pred_sem2')

        f, [[ax0, ax1, ax2], [ax3, ax4, ax5]] = plt.subplots(2,3,figsize=(16,4))

        ax0.imshow(visualize_semantic_processed(pred_sem1, [4,10,18]))
        ax3.imshow(rgb1)
        ax1.imshow(visualize_semantic_processed(pred_sem2, [4,10,18]))
        ax4.imshow(rgb2)
        ax5.bar(['pred', 'gt'], [pred_bra, bra])

        opt_info.update({'it': it, 'viz': wandb.Image(plt)})
        wandb.log(opt_info)
        plt.close('all')

    def log_rgb_info(self, it, opt_info):

        rgb = opt_info.pop('rgb')
        bev = opt_info.pop('bev')
        sem = opt_info.pop('sem')
        pred_sem = opt_info.pop('pred_sem')

        locs = opt_info.pop('locs')
        pred_locs = opt_info.pop('pred_locs')
        nxp = opt_info.pop('nxp')

        cmd = {0:'left',1:'right',2:'straight',3:'follow',4:'change left',5:'change right'}.get(opt_info.pop('cmd'))

        f, [[ax1,ax2], [ax3,ax4]] = plt.subplots(2,2,figsize=(8,8))

        ax1.imshow(bev, cmap='gray')
        ax2.imshow(rgb)
        ax3.imshow(visualize_semantic_processed(sem))
        ax4.imshow(visualize_semantic_processed(pred_sem))

        for loc_x, loc_y in locs:
            ax1.add_patch(Circle((loc_x,loc_y),radius=0.5, color=RED))
        
        for pred_loc in pred_locs:
            for loc_x, loc_y in pred_loc:
                ax1.add_patch(Circle((loc_x,loc_y),radius=0.5, color=GREEN))

        ax1.add_patch(Circle(nxp,radius=0.5, color=BLUE))

        opt_info.update({'it': it, 'viz': wandb.Image(plt)})
        wandb.log(opt_info)
        plt.close('all')


    def log_lidar_info(self, it, opt_info):

        dets = opt_info.pop('det')
        gt_dets = opt_info.pop('gt_det')

        bev = opt_info.pop('pred_bev')
        gt_bev = opt_info.pop('bev')

        other_next_locs = opt_info.pop('other_next_locs')
        other_cast_locs = opt_info.pop('other_cast_locs')
        other_cast_cmds = opt_info.pop('other_cast_cmds')
        ego_plan_locs = opt_info.pop('ego_plan_locs')
        ego_next_locs = opt_info.pop('ego_next_locs')
        
        nxp = opt_info.pop('nxp')

        num_points = opt_info.pop('num_points')

        cmd = {0:'left',1:'right',2:'straight',3:'follow',4:'change left',5:'change right'}.get(opt_info.pop('cmd'))

        f, [ax1, ax2] = plt.subplots(1,2,figsize=(8,4))

        ax1.imshow(gt_bev, cmap='gray')
        ax1.set_title(cmd)
        ax2.imshow(bev, cmap='gray')

        for color, det in zip([ORANGE, RED], gt_dets):
            for x, y, w, h, cos, sin in det:
                ax1.add_patch(Rectangle((x,y)+[w,h]@np.array([[-sin,cos],[-cos,-sin]]), w*2, h*2, angle=np.rad2deg(np.arctan2(sin, cos)-np.pi/2), color=color))
                ax1.add_patch(Arrow(x,y,ARROW_WIDTH*sin,-ARROW_WIDTH*cos,color=BLACK,width=ARROW_WIDTH))
                # ax1.add_patch(Rectangle((x,y)-(h*3/2,w/2)@np.array([[-cos,sin],[-sin,-cos]]), w*2, h*2, angle=90+np.rad2deg(np.arctan2(cos, sin)), color=color))

        for color, det in zip([ORANGE, RED], dets):
            for x, y, w, h, cos, sin in det:
                ax2.add_patch(Rectangle((x,y)+[w,h]@np.array([[-sin,cos],[-cos,-sin]]), w*2, h*2, angle=np.rad2deg(np.arctan2(sin, cos)-np.pi/2), color=color))
                ax2.add_patch(Arrow(x,y,ARROW_WIDTH*sin,-ARROW_WIDTH*cos,color=BLACK,width=ARROW_WIDTH))
                # ax2.add_patch(Rectangle((x,y)-(h*3/2,w/2)@np.array([[-cos,sin],[-sin,-cos]]), w*2, h*2, angle=90+np.rad2deg(np.arctan2(cos, sin)), color=color))

        for loc_x, loc_y in ego_next_locs:
            ax1.add_patch(Circle((loc_x,loc_y),radius=0.5, color=RED))

        for loc_x, loc_y in ego_plan_locs:
            ax2.add_patch(Circle((loc_x,loc_y),radius=0.5, color=GREEN))

        ax1.add_patch(Circle(nxp,radius=1, color=ORANGE))

        cmap = matplotlib.cm.get_cmap('jet')
        for scores, trajs in zip(other_cast_cmds, other_cast_locs):
            for i in range(self.num_cmds):
                score = scores[i]
                traj  = trajs[i]
                # if score < self.cmd_thresh:
                #     continue

                for loc_x, loc_y in traj:
                    ax2.add_patch(Circle((loc_x,loc_y),radius=0.5, color=cmap(score)))

        for other_next_loc in other_next_locs:
            for loc_x, loc_y in other_next_loc:
                ax1.add_patch(Circle((loc_x,loc_y),radius=0.5, color=RED))

        ax1.set_xticks([])
        ax2.set_xticks([])
        ax1.set_yticks([])
        ax2.set_yticks([])

        opt_info.update({'it': it, 'viz': wandb.Image(plt), 'num_points': wandb.Histogram(num_points)})
        wandb.log(opt_info)
        plt.close('all')


    def log_seg_info(self, it, opt_info):

        rgb = opt_info.pop('rgb')
        sem = opt_info.pop('sem')
        pred_sem = opt_info.pop('pred_sem')

        f, [ax1, ax2, ax3] = plt.subplots(1,3,figsize=(12,4))

        ax1.imshow(rgb)
        ax2.imshow(visualize_semantic_processed(sem))
        ax3.imshow(visualize_semantic_processed(pred_sem))

        opt_info.update({'it': it, 'viz': wandb.Image(plt)})
        wandb.log(opt_info)
        plt.close('all')

    def log_bev_info(self, it, opt_info):

        bev           = opt_info.pop('bev')
        nxp           = opt_info.pop('nxp')
        ego_plan_locs = opt_info.pop('ego_plan_locs')
        ego_cast_locs = opt_info.pop('ego_cast_locs')
        ego_cast_cmds = opt_info.pop('ego_cast_cmds')

        h, w = bev.shape

        cmd = {0:'left',1:'right',2:'straight',3:'follow',4:'change left',5:'change right'}.get(opt_info.pop('cmd'))

        f, ax = plt.subplots(1,1,figsize=(4,4))
        ax.imshow(bev, cmap='gray')
        ax.set_title(cmd)

        cmap = matplotlib.cm.get_cmap('jet')

        ax.add_patch(Circle(np.clip(nxp, 0, w),radius=1, color=ORANGE))

        for score, ego_cast_loc in zip(ego_cast_cmds, ego_cast_locs):
            for loc_x, loc_y in ego_cast_loc:
                ax.add_patch(Circle((loc_x,loc_y),radius=0.5, color=cmap(score)))

        for loc_x, loc_y in ego_plan_locs:
            ax.add_patch(Circle((loc_x,loc_y),radius=0.5, color=ORANGE))

        opt_info.update({'it': it, 'viz': wandb.Image(plt)})
        wandb.log(opt_info)
        plt.close('all')

    def save(self, paths):
        for path in paths:
            wandb.save(path)
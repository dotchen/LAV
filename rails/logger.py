import yaml
import math
import ray
import wandb
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.visualization import visualize_semantic_processed

class Logger:
    def __init__(self, wandb_project, args):
        
        # Read configs
        with open(args.config_path, 'rb') as f:
            config = yaml.safe_load(f)
            
        for key, value in config.items():
            setattr(self, key, value)

        wandb.init(project=wandb_project, config=config)
    
    @property
    def save_dir(self):
        return wandb.run.dir
        
    def log_ego_info(self, it, opt_info):
        
        width = 0.1
        length = 0.5

        locs = opt_info.pop('locs')
        yaws = opt_info.pop('yaws')
        pred_locs = opt_info.pop('pred_locs')
        pred_yaws = opt_info.pop('pred_yaws')
        
        # Normalize the locations
        pred_locs = pred_locs - locs[:,0:1]
        locs = locs - locs[:,0:1]
        
        f, axes = plt.subplots(2,2,figsize=(10,10))
        
        for i, ax in enumerate(itertools.chain(*axes)):
            ax.set_xlim([-10,10])
            ax.set_ylim([-10,10])
            for loc, yaw, pred_loc, pred_yaw in zip(locs[i,1:], yaws[i,1:], pred_locs[i], pred_yaws[i]):
                ax.arrow(*loc,length*math.cos(yaw),length*math.sin(yaw), color='blue', width=width)
                ax.arrow(*pred_loc,length*math.cos(pred_yaw),length*math.sin(pred_yaw), color='red', width=width)
        
        opt_info.update({'it': it, 'viz': wandb.Image(plt)})
        wandb.log(opt_info)
        plt.close('all')
        
    def log_main_info(self, it, opt_info):
        
        wide_rgb = opt_info.pop('wide_rgb')
        narr_rgb = opt_info.pop('narr_rgb')
        spd      = opt_info.pop('spd')
        cmd      = opt_info.pop('cmd')
        pred_seg = opt_info.pop('pred_seg')
        gt_seg   = opt_info.pop('gt_seg')
        act_prob = opt_info.pop('act_prob')
        act_brak = opt_info.pop('act_brak')
        pred_act_prob = opt_info.pop('pred_act_prob')
        pred_act_brak = opt_info.pop('pred_act_brak')

        pred_seg = visualize_semantic_processed(pred_seg, self.seg_channels)
        gt_seg   = visualize_semantic_processed(gt_seg, self.seg_channels)
        
        f, [[ax1,ax2,ax3], [ax4,ax5,ax6]] = plt.subplots(2,3,figsize=(30,10))

        ax1.imshow(narr_rgb); 
        ax4.imshow(wide_rgb);           ax4.set_title({0:'Left',1:'Right',2:'Straight',3:'Follow',4:'Change Left',5:'Change Right'}.get(cmd))
        ax2.imshow(pred_seg);           ax2.set_title('predicted sem')
        ax5.imshow(gt_seg);             ax5.set_title('gt sem')
        ax3.imshow(pred_act_prob);      ax3.set_title(f'(pred) brake: {pred_act_brak:.3f}')
        ax6.imshow(act_prob);           ax6.set_title(f'(gt) brake: {act_brak:.3f}')

        opt_info.update({'it': it, 'viz': wandb.Image(plt)})
        wandb.log(opt_info)
        plt.close('all')


    def log_label_info(self, label_info):

        act_val_norm = label_info.pop('act_val_norm')
        act_val_brak = label_info.pop('act_val_brak')
        cmd = label_info.pop('cmd')
        wide_rgb = label_info.pop('wide_rgb')

        f, [ax1, ax2] = plt.subplots(1,2,figsize=(30,10))

        ax1.imshow(wide_rgb)
        ax2.imshow(act_val_norm)
        ax1.set_title({0:'Left',1:'Right',2:'Straight',3:'Follow',4:'Change left',5:'Change right'}.get(cmd))
        ax2.set_title(act_val_brak)

        wandb.log({'viz': wandb.Image(plt)})
        plt.close('all')
    
   
@ray.remote
class RemoteLogger(Logger):
    def __init__(self, wandb_project, config):
        super().__init__(wandb_project, config)
        self.worker_counts = defaultdict(lambda: 0)
        
    def log_label_info(self, label_info, worker_count, worker_id=0):
        
        super().log_label_info(label_info)
        
        self.worker_counts[worker_id] = worker_count
        
    def total_frames(self):
        return sum(self.worker_counts.values())

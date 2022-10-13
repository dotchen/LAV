import tqdm
import numpy as np
import torch
from .wor import WOR
from .datasets import data_loader
from .logger import Logger

def main(args):
    
    wor = WOR(args)
    data = data_loader('main', args)
    logger = Logger('carla_train_phase2', args)
    save_dir = logger.save_dir
    
    if args.resume:
        print ("Loading checkpoint from", args.resume)
        if wor.multi_gpu:
            wor.main_model.module.load_state_dict(torch.load(args.resume))
        else:
            wor.main_model.load_state_dict(torch.load(args.resume))
        start = int(args.resume.split('main_model_')[-1].split('.th')[0])
    else:
        start = 0

    global_it = 0
    for epoch in range(start,start+args.num_epoch):
        for wide_rgbs, wide_sems, narr_rgbs, narr_sems, act_vals, spds, cmds in tqdm.tqdm(data, desc='Epoch {}'.format(epoch)):
            opt_info = wor.train_main(wide_rgbs, wide_sems, narr_rgbs, narr_sems, act_vals, spds, cmds)
            
            if global_it % args.num_per_log == 0:
                logger.log_main_info(global_it, opt_info)
        
            global_it += 1
    
        # Save model
        if (epoch+1) % args.num_per_save == 0:
            save_path = f'{save_dir}/main_model_{epoch+1}.th'
            torch.save(wor.main_model_state_dict(), save_path)
            print (f'saved to {save_path}')

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--resume', default=None)
    
    parser.add_argument('--data-dir', default='/ssd2/dian/challenge_data/main_trajs6')
    parser.add_argument('--config-path', default='/home/dianchen/carla_challenge/config_wor.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    
    # Training data config
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--num-repeat', type=int, default=4)    # Should be consistent with autoagents/collector_agents/config.yaml

    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=3e-5)
    
    parser.add_argument('--num-per-log', type=int, default=100, help='per iter')
    parser.add_argument('--num-per-save', type=int, default=1, help='per epoch')
    
    parser.add_argument('--balanced-cmd', action='store_true')

    args = parser.parse_args()
    main(args)
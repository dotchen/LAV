import tqdm
import torch
from .wor import WOR
from .datasets import data_loader
from .logger import Logger


def main(args):
    
    # WAR!!
    wor = WOR(args)
    data = data_loader('ego', args)
    logger = Logger('carla_train_phase0', args)
    
    global_it = 0
    for epoch in tqdm.tqdm(range(args.num_epoch)):
        for locs, rots, spds, acts in data: 
            opt_info = wor.train_ego(locs, rots, spds, acts)
            
            if global_it % args.num_per_log == 0:
                logger.log_ego_info(global_it, opt_info)
            
            global_it += 1

    # Save model
    torch.save(wor.ego_model.state_dict(), 'ego_model.th')


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-dir', default='/ssd2/dian/challenge_data/ego_trajs')
    parser.add_argument('--config-path', default='/home/dianchen/carla_challenge/config.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')

    # Training data config
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--num-repeat', type=int, default=4)    # Should be consistent with autoagents/collector_agents/config.yaml

    parser.add_argument('--num-workers', type=int, default=0) 
    parser.add_argument('--ego-traj-len', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)

    # Logging config
    parser.add_argument('--num-per-log', type=int, default=10)

    args = parser.parse_args()

    main(args)

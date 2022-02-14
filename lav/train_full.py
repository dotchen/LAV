import tqdm
import torch
from lav.lav_final import LAV
from lav.utils.datasets import get_data_loader
from lav.utils.logger import Logger

def main(args):

    lav = LAV(args)
    data_loader = get_data_loader('lidar_painted' if lav.point_painting else 'lidar', args)
    logger = Logger('lav_lidar', args)
    save_dir = logger.save_dir

    torch.manual_seed(args.seed)

    logger.watch_model(lav.lidar_model)
    logger.watch_model(lav.uniplanner)

    global_it = 0
    for epoch in range(args.num_epoch):
        for data in tqdm.tqdm(data_loader, desc=f'Epoch {epoch}'):

            opt_info = lav.train_lidar(*data)

            if global_it % args.num_per_log == 0:
                logger.log_lidar_info(global_it, opt_info)

            global_it += 1
        
        lav.lidar_scheduler.step()

        # Save model
        if (epoch+1) % args.num_per_save == 0:
            lidar_path = f'{save_dir}/lidar_{epoch+1}.th'
            torch.save(lav.state_dict('lidar'), lidar_path)
            print (f'saved to {lidar_path}')

            uniplanner_path = f'{save_dir}/uniplanner_{epoch+1}.th'
            torch.save(lav.state_dict('uniplanner'), uniplanner_path)
            print (f'saved to {uniplanner_path}')

            logger.save([lidar_path, uniplanner_path])



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', default='config.yaml')

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    parser.add_argument('--perceive-only', action='store_true')

    # Training misc
    parser.add_argument('--num-epoch', type=int, default=64)
    parser.add_argument('--num-per-log', type=int, default=100, help='log per iter')
    parser.add_argument('--num-per-save', type=int, default=1, help='save per epoch')

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-workers', type=int, default=16)

    # Reproducibility
    parser.add_argument('--seed', type=int, default=2021)

    args = parser.parse_args()

    main(args)
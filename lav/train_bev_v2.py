import tqdm
import torch
from lav.lav_privileged_v2 import LAV
from lav.utils.datasets import get_data_loader
from lav.utils.logger import Logger

def main(args):

    dmd = LAV(args)
    data_loader = get_data_loader('temporal_bev', args)
    logger = Logger('lav_bev', args)
    save_dir = logger.save_dir

    torch.manual_seed(args.seed)

    global_it = 0
    for epoch in range(args.num_epoch):
        for data in tqdm.tqdm(data_loader, desc=f'Epoch {epoch}'):

            opt_info = dmd.train_bev(*data, other_weight=other_weight(global_it))

            if global_it % args.num_per_log == 0:
                logger.log_bev_info(global_it, opt_info)

            global_it += 1

        dmd.bev_scheduler.step()

        if (epoch+1) % args.num_per_save == 0:
            bev_path = f'{save_dir}/bev_{epoch+1}.th'
            torch.save(dmd.state_dict('bev'), bev_path)
            print (f'save to {bev_path}')

            logger.save([bev_path])

def other_weight(it,beta=0.8):
    return 1-beta**(it/4000)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', default='config_v2.yaml')

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    # Training misc
    parser.add_argument('--num-epoch', type=int, default=160)
    parser.add_argument('--num-per-log', type=int, default=100, help='log per iter')
    parser.add_argument('--num-per-save', type=int, default=1, help='save per epoch')

    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument('--num-workers', type=int, default=16)

    # Reproducibility (still not fully determinstic due to CUDA/CuDNN)
    parser.add_argument('--seed', type=int, default=2021)

    args = parser.parse_args()

    main(args)

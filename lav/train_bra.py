import tqdm
import torch
from lav.lav_privileged import LAV
from lav.utils.datasets import get_data_loader
from lav.utils.logger import Logger

def main(args):
    
    dmd = LAV(args)
    data_loader = get_data_loader('bra', args)
    logger = Logger('lav_bra', args)
    save_dir = logger.save_dir

    torch.manual_seed(args.seed)

    global_it = 0
    for epoch in range(args.num_epoch):
        for rgb1, rgb2, sem1, sem2, bra in tqdm.tqdm(data_loader, desc=f'Epoch {epoch}'):
            opt_info = dmd.train_bra(rgb1, rgb2, sem1, sem2, bra)

            if global_it % args.num_per_log == 0:
                logger.log_bra_info(global_it, opt_info)

            global_it += 1

        # Save model
        if (epoch+1) % args.num_per_save == 0:
            bra_path = f'{save_dir}/bra_{epoch+1}.th'

            torch.save(dmd.state_dict('bra'), bra_path)
            print (f'saved to {bra_path}')

            logger.save([bra_path])


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', default='config.yaml')

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    # Training misc
    parser.add_argument('--num-epoch', type=int, default=10)
    parser.add_argument('--num-per-log', type=int, default=100, help='log per iter')
    parser.add_argument('--num-per-save', type=int, default=1, help='save per epoch')
    
    parser.add_argument('--batch-size', type=int, default=52)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-workers', type=int, default=16)
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=2021)

    args = parser.parse_args()
    
    main(args)
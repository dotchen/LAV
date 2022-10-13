import ray
import time
import tqdm
import torch
from .wor import WORActionLabeler
from .datasets.main_dataset import RemoteMainDataset
from .logger import RemoteLogger

def main(args):
    
    # logger = RemoteLogger.remote('carla_data_phase2', args)
    dataset = RemoteMainDataset.remote(args.data_dir, args.config_path)
    total_frames = ray.get(dataset.num_frames.remote())

    jobs = []
    for worker_id in range(args.num_workers):
        labeler = WORActionLabeler.remote(args, dataset, worker_id=worker_id, total_worker=args.num_workers)
        jobs.append(labeler.run.remote(logger))

    frames = 0
    pbar = tqdm.tqdm(total=total_frames)
    while True:
        time.sleep(1.)
        current_frames = ray.get(logger.total_frames.remote())
        pbar.update(current_frames - frames)
        frames = current_frames
        
        if frames >= total_frames:
            break

    ray.wait([dataset.commit.remote()])

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='/ssd2/dian/challenge_data/main_trajs6')
    parser.add_argument('--config-path', default='config.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')

    # Training data config
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--num-repeat', type=int, default=4)    # Should be consistent with autoagents/collector_agents/config.yaml

    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--ego-traj-len', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)

    # Logging config
    parser.add_argument('--num-per-log', type=int, default=100)

    args = parser.parse_args()
    
    ray.init(logging_level=30, local_mode=False, log_to_driver=True)
    main(args)

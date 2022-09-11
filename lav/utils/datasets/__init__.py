from .bev_dataset import BEVDataset
from .rgb_dataset import RGBDataset
from .seg_dataset import SegmentationDataset
from .bra_dataset import BrakePredictionDataset
from .lidar_dataset import LiDARDataset
from .lidar_painted_dataset import LiDARPaintedDataset
from .temporal_bev_dataset import TemporalBEVDataset
from .temporal_lidar_painted_dataset import TemporalLiDARPaintedDataset
from torch.utils.data import DataLoader


def get_data_loader(data_type, args):

    if data_type == 'bev':
        dataset_cls = BEVDataset
    elif data_type == 'temporal_bev':
        dataset_cls = TemporalBEVDataset
    elif data_type == 'rgb':
        dataset_cls = RGBDataset
    elif data_type == 'seg':
        dataset_cls = SegmentationDataset
    elif data_type == 'bra':
        dataset_cls = BrakePredictionDataset
    elif data_type == 'lidar':
        dataset_cls = LiDARDataset
    elif data_type == 'lidar_painted':
        dataset_cls = LiDARPaintedDataset
    elif data_type == 'temporal_lidar_painted':
        dataset_cls = TemporalLiDARPaintedDataset
    else:
        raise NotImplementedError

    return DataLoader(
        dataset_cls(args.config_path, seed=args.seed),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

__all__ = ['get_data_loader']

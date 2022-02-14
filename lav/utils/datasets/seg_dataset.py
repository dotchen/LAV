import numpy as np
from .basic_dataset import BasicDataset
from lav.utils.augmenter import augment
from lav.utils import filter_sem


class SegmentationDataset(BasicDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.augmenter = augment(0.5)

    def __len__(self):
        return self.num_frames * len(self.camera_yaws)

    def __getitem__(self, idx):

        frame_index  = idx // len(self.camera_yaws)
        camera_index = idx %  len(self.camera_yaws)

        lmdb_txn = self.txn_map[frame_index]
        index = self.idx_map[frame_index]

        rgb = self.__class__.load_img(lmdb_txn, 'rgb_{}'.format(camera_index), index)
        sem = self.__class__.load_img(lmdb_txn, 'sem_{}'.format(camera_index), index)

        rgb = self.augmenter(images=rgb[...,::-1][None])[0]

        sem = filter_sem(sem, self.seg_channels)
        
        return rgb, sem

if __name__ == '__main__':
    dataset = SegmentationDataset('config.yaml')

    import tqdm
    for t in tqdm.tqdm(range(200)):
        dataset[t]
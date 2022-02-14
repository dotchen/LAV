import numpy as np
from .basic_dataset import BasicDataset
from lav.utils.augmenter import augment
from lav.utils import filter_sem

class BrakePredictionDataset(BasicDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.augmenter = augment(0.5)

    def __getitem__(self, idx):

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]
        
        rgb1 = self.__class__.load_img(lmdb_txn, 'rgb_{}'.format(int(len(self.camera_yaws)/2)-1), index)
        rgb2 = self.__class__.load_img(lmdb_txn, 'rgb_{}'.format(int(len(self.camera_yaws)/2)), index)
        rgb3 = self.__class__.load_img(lmdb_txn, 'rgb_{}'.format(int(len(self.camera_yaws)/2)+1), index)
        tel_rgb = self.__class__.load_img(lmdb_txn, 'tel_rgb', index)


        sem1 = self.__class__.load_img(lmdb_txn, 'sem_{}'.format(int(len(self.camera_yaws)/2)-1), index)
        sem2 = self.__class__.load_img(lmdb_txn, 'sem_{}'.format(int(len(self.camera_yaws)/2)), index)
        sem3 = self.__class__.load_img(lmdb_txn, 'sem_{}'.format(int(len(self.camera_yaws)/2)+1), index)
        tel_sem = self.__class__.load_img(lmdb_txn, 'tel_sem', index)

        bra = int(self.__class__.access('bra', lmdb_txn, index, 1, dtype=np.uint8))
        
        rgb = np.concatenate([rgb1, rgb2, rgb3], axis=1)
        rgb = self.augmenter(images=rgb[...,::-1][None])[0]
        tel_rgb = tel_rgb[:-self.crop_tel_bottom]
        tel_rgb = self.augmenter(images=tel_rgb[...,::-1][None])[0]

        sem = np.concatenate([sem1, sem2, sem3], axis=1)
        sem = filter_sem(sem, [4,10,18])
        tel_sem = filter_sem(tel_sem, [4,10,18])
        tel_sem = tel_sem[:-self.crop_tel_bottom]

        return rgb, tel_rgb, sem, tel_sem, bra


if __name__ == '__main__':
    dataset = BrakePredictionDataset('config.yaml')

    import tqdm
    for t in tqdm.tqdm(range(200)):
        dataset[t]

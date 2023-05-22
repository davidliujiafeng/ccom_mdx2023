from os import path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from loader.bleeding import WaveSetBleeding


class WaveBleedingDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.root_dir = args.root_dir
        self.meta_dir = args.meta_dir
        self.seg_len = args.seg_len
        self.shift = args.shift
        self.batch_size = args.batch_size
        self.meta_train = args.meta_train

    def setup(self, stage: str = None) -> None:
        self.train_set = WaveSetBleeding(self.root_dir,
                                 path.join(self.meta_dir, self.meta_train),
                                 seg_len=self.seg_len,
                                 shift=self.shift)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=64, drop_last=True,
                          pin_memory=True)


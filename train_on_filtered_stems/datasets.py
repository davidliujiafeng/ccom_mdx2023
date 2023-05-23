import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from os import path

import torch
import torchaudio
import tqdm
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from augment import change_pitch_tempo

import random

class WaveLBSet(Dataset):
    def __init__(self, root, metadata_path,
                 seg_len=44100 * 11, shift=44100, sample_rate=44100, channels=2):
        self.root = root
        self.metadata = OrderedDict(json.load(open(metadata_path)))
        self.sources = ["drums.wav", "bass.wav", "other.wav", "vocals.wav"]
        self.seg_len = seg_len
        self.shift = shift
        self.channels = channels
        self.sample_rate = sample_rate
        self.data_idx = {}

        # Create index for training data
        for source, wavs in self.metadata.items():
            # print(source, wavs)
            self.data_idx[source] = []
            for file_name, info in wavs.items():
                # print(file_name, info)
                for i in range(0, info["length"] - seg_len + 1, shift):
                    self.data_idx[source].append((file_name, i))

    def __len__(self):
        return 2048 * 8 * 2 # 2048 samples for 1 epoch
    
    def __getitem__(self, index):

        wavs = []
        for source in self.sources:
            source = source.split(".")[0]
            rand_idx = random.randint(0, len(self.data_idx[source]) - 1)

            file_name, start_idx = self.data_idx[source][rand_idx]
            file_path = self.root + "/" + source + "/" + file_name

            wav = change_pitch_tempo(file_path, start_idx, self.seg_len)
            wavs.append(wav)

        one_sample = torch.stack(wavs)

        return one_sample
     
        
class WaveLabelDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.root_dir = args.root_dir
        self.meta_dir = args.meta_dir
        self.seg_len = args.seg_len
        self.shift = args.shift
        self.batch_size = args.batch_size
        self.meta_train = args.meta_train

    def setup(self, stage: str = None) -> None:
        self.train_set = WaveLBSet(self.root_dir,
                                 path.join(self.meta_dir, self.meta_train),
                                 seg_len=self.seg_len,
                                 shift=self.shift)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=64, drop_last=True,
                          pin_memory=True)




import json
from collections import OrderedDict

import torch
import torchaudio
from torch.utils.data import Dataset
from augment import change_pitch_tempo
import random


# class WaveSetBleeding(Dataset):
#     def __init__(self, root, metadata_path,
#                  seg_len=44100 * 11, shift=44100, sample_rate=44100, channels=2):
#         self.root = root
#         self.metadata = OrderedDict(json.load(open(metadata_path)))
#         self.sources = ["drums.wav", "bass.wav", "other.wav", "vocals.wav"]
#         self.seg_len = seg_len
#         self.shift = shift
#         self.channels = channels
#         self.sample_rate = sample_rate
#         self.data_idx = []
#
#         # Create index for training data
#         for file_name, info in self.metadata.items():
#             '''
#             Calculate the starting point of each segment
#             self.data_idx looks like this:
#
#             [('song1', 0), ('song1', 44100), ('song1', 88200)....('song6', 13230).....]
#
#             The length of self.data_idx is the number of total samples
#             '''
#             for i in range(0, info["length"] - seg_len + 1, shift):
#                 self.data_idx.append((file_name, i))
#
#     def __len__(self):
#         return len(self.data_idx)
#
#     def __getitem__(self, index):
#         wavs = []
#
#         for source in self.sources:
#             file_path = self.root + "/" + self.data_idx[index][0] + "/" + source
#             start_idx = self.data_idx[index][1]
#             wav = change_pitch_tempo(file_path, start_idx, self.seg_len)  # Augmentation
#             # wav, _ = torchaudio.load(str(file_path), frame_offset=start_idx, num_frames=self.seg_len)
#             wavs.append(wav)
#         one_sample = torch.stack(wavs)  # [4, 2, 485100]
#
#         return one_sample


class WaveSetBleeding(Dataset):
    def __init__(self, root, metadata_path,
                 seg_len=44100 * 11, shift=44100, sample_rate=44100, channels=2):
        self.root = root
        self.metadata = OrderedDict(json.load(open(metadata_path)))
        self.sources = ["drums.wav", "bass.wav", "other.wav", "vocals.wav"]
        self.seg_len = seg_len
        self.shift = shift
        self.channels = channels
        self.sample_rate = sample_rate
        self.data_idx = []

        # Create index for training data
        for file_name, info in self.metadata.items():
            '''
            Calculate the starting point of each segment
            self.data_idx looks like this:

            [('song1', 0), ('song1', 44100), ('song1', 88200)....('song6', 13230).....]

            The length of self.data_idx is the number of total samples
            '''
            for i in range(0, info["length"] - seg_len + 1, shift):
                self.data_idx.append((file_name, i))

    def __len__(self):
        return 1024 * 8 * 4

    def __getitem__(self, index):
        wavs = []

        for source in self.sources:
            rand_idx = random.randint(0, len(self.data_idx) - 1)

            file_path = self.root + "/" + self.data_idx[rand_idx][0] + "/" + source
            start_idx = self.data_idx[rand_idx][1]
            wav = change_pitch_tempo(file_path, start_idx, self.seg_len)  # Augmentation
            # wav, _ = torchaudio.load(str(file_path), frame_offset=start_idx, num_frames=self.seg_len)
            wavs.append(wav)
        one_sample = torch.stack(wavs)  # [4, 2, 485100]

        return one_sample


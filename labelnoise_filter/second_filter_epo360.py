
from pathlib import Path
import time
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_on_filtered_stems'))
from module import LitModuleWrapper
def separate_simple(model, mix, seg_len, device):
    """
    This is a naive method to separate audio without overlap
    """
    num_missing = seg_len - (mix.shape[-1] % seg_len)
    mix = F.pad(mix, (0, num_missing))
    chunks = torch.split(mix, seg_len, dim=-1)
    estimates = []
    for chunk in chunks:
        chunk = chunk.to(device)
        estimate_ = model(chunk)
        estimates.append(estimate_.detach().cpu())
        del estimate_, chunk

    estimate = torch.cat(estimates, dim=-1)
    estimate = estimate[..., :-num_missing]

    return estimate

class MyDemucsWrapper:
    """
    Demucs model for music demixing.
    """
    def __init__(self):

        my_args = OmegaConf.load('../config/labelnoise.yaml')
        self.separator = LitModuleWrapper(my_args)
        self.separator.model.load_state_dict(torch.load('checkpoints/epoch=359-on_filtered.th'))
        self.separator.freeze()

        if torch.cuda.device_count():
            self.device = torch.device('cuda')
        else:
            print("WARNING, using CPU")
            self.device = torch.device('cpu')
        self.separator.to(self.device)
    @property
    def instruments(self):
        """ DO NOT CHANGE """
        return ['bass', 'drums', 'other', 'vocals']

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)

    def separate_music_file(self, mixed_sound_array, sample_rate):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate

        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """

        # create input for Demucs model
        mix = torch.from_numpy(np.asarray(mixed_sound_array, np.float32).T)

        mix_channels = mix.shape[0]

        # # Normalize track, no required for any recent version of Demucs but never hurts.
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std
        # Separate
        with torch.no_grad():
            out_length = int(self.separator.args.dset.segment * 38808 - 44100)
            estimates = separate_simple(self.separator, mix.unsqueeze(0), out_length, self.device)
            estimates = estimates.squeeze(0)

        estimates = estimates * std + mean

        separated_music_arrays = {}
        output_sample_rates = {}
        for instrument in self.instruments:
            idx = self.separator.args.dset.sources.index(instrument)
            separated_music_arrays[instrument] = estimates[idx].numpy().T
            output_sample_rates[instrument] = sample_rate

        return separated_music_arrays, output_sample_rates

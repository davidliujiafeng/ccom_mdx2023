
from pathlib import Path
import time
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf

from my_submission.pl_demucs_module import LitModuleWrapper, sep_weighted_avg_shift
import torch
import sys

class MyDemucsWrapper:
    """
    Demucs model for music demixing.
    """
    def __init__(self):

        my_args = OmegaConf.load('my_submission/config.yaml')
        self.separator = LitModuleWrapper(my_args)
        self.separator.model_bag['drums'].load_state_dict(torch.load('checkpoints/epoch=443_dbv.th'))
        self.separator.model_bag['bass'].load_state_dict(torch.load('checkpoints/epoch=1003_bv.th'))
        self.separator.model_bag['other'].load_state_dict(torch.load('checkpoints/epoch=879_o.th'))
        self.separator.model_bag['vocals'].load_state_dict(torch.load('checkpoints/epoch=1142_v.th'))
        self.separator.freeze()

        # we select automatically the device based on what is available,
        # remember to change in aicrowd.json if you want to use the GPU or not.
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
        mix = mix.to(self.device)
        
        # convert audio to GPU
        # print(mix.shape)
        mix_channels = mix.shape[0]

        # # Normalize track, no required for any recent version of Demucs but never hurts.
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std

        estimates = dict()
        with torch.no_grad():
            out_length = int(self.separator.args.dset.segment * 38808 - 44100)
            for source in self.separator.args.dset.sources:
                ov = 0.25
                sh = 1
                if source == 'drums':
                    ov = 0.75
                    sh = 1
                if source == 'bass':
                    ov = 0.75
                    sh = 1
                estimates[source] = sep_weighted_avg_shift(self.separator.model_bag[source], mix.unsqueeze(0), out_length, overlap=ov, shifts=sh)
                estimates[source] = estimates[source].squeeze(0) * std + mean


        separated_music_arrays = {}
        output_sample_rates = {}
        for instrument in self.instruments:
            idx = self.separator.args.dset.sources.index(instrument)
            separated_music_arrays[instrument] = estimates[self.separator.args.dset.sources[idx]][idx].detach().cpu().numpy().T
            output_sample_rates[instrument] = sample_rate

        return separated_music_arrays, output_sample_rates

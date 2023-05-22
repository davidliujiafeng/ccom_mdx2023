# This file uses Hybrid Demucs for music demixing.
# It is one of official baselines for the Music Demixing challenge.
#
# Reference: Alexandre Défossez. "Hybrid Spectrogram and Waveform Source Separation"
#            MDX Workshop at ISMIR 2021
#
# NOTE:
# a) Demucs needs checkpoints to be submitted along with your code.
# b) Please upgrade Demucs to the latest release (4.0.0).
#
# If you trained your model with the Demucs codebase, make sure to export
# your model, using e.g. `python -m tools.export SIG`. Then copy the files
# `release_models/SIG.th` into this repo.
# Update the SIG in the get_model hereafter.
#
# /!\ Remember to update the aicrowd.json to match your use case.
#
# Making submission using demucs:
# 2. Run this file locally with `python evaluate_locally.py`.
# 4. Submit your code using git-lfs
#    #> git lfs install
#    #> git lfs track "*.th"
#    #> git add .gitattributes
#    #> git add models
#    #> git add -u .

# Follow the instructions in the docs/submission.md file.
# Once the repo is properly setup, you can easily push new submissions with
# > git add models; git add -u .
# > git commit -m "commit message"
# > name="submission name here" ; git tag -am "submission-$name" submission-$name; git push aicrowd submission-$name


from pathlib import Path
import time
import numpy as np
import torch
from my_submission.separate import sep_weighted_avg, sep_weighted_avg_shift
from my_submission.MultiSUnet_v2 import MultiSUnet_v2, MultiSUnet_v2_bass


class MyWrapperBag:
    def __init__(self):
        # # -----------------------------------------------------------------------------------------
        self.separator_bass = MultiSUnet_v2_bass(num_sources=1, n_ffts=[16384])
        checkpoint_bass = torch.load('models/MultiSUnet_v2_d5_c32_ep203_bass_byv13.th')
        self.separator_bass.load_state_dict(checkpoint_bass['model_state_dict'])
        # # -----------------------------------------------------------------------------------------
        self.separator_other = MultiSUnet_v2(num_sources=4, n_ffts=[4096, 8192, 16384])
        checkpoint_other = torch.load('models/MultiSUnet_v2_d5_c32_ep364_4S_v44.th')
        self.separator_other.load_state_dict(checkpoint_other['model_state_dict'])
        # -----------------------------------------------------------------------------------------
        self.separator_vocal = MultiSUnet_v2()
        checkpoint_vocal = torch.load('models/MultiSUnet_v2_d5_c32_ep430_vocal.th')
        self.separator_vocal.load_state_dict(checkpoint_vocal['model_state_dict'])
        # -----------------------------------------------------------------------------------------
        # self.separator_drum.eval()
        self.separator_bass.eval()
        self.separator_other.eval()
        self.separator_vocal.eval()

        self.sources = ['drums', 'bass', 'other', 'vocals']

        # we select automatically the device based on what is available,
        # remember to change in aicrowd.json if you want to use the GPU or not.
        if torch.cuda.device_count():
            self.device = torch.device('cuda')
        else:
            print("WARNING, using CPU")
            self.device = torch.device('cpu')

        # self.separator_drum.to(self.device)
        self.separator_bass.to(self.device)
        self.separator_other.to(self.device)
        self.separator_vocal.to(self.device)

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
        mix = torch.from_numpy(np.asarray(mixed_sound_array.T, np.float32))
        mix = mix.unsqueeze(0)

        # convert audio to GPU
        mix = mix.to(self.device)

        # Separate
        with torch.no_grad():
            out_length = 1024 * 255
            estimate_bass = sep_weighted_avg_shift(self.separator_bass, mix, out_length, overlap=0.75, shifts=4)

        with torch.no_grad():
            out_length = 1024 * 255
            estimate_other = sep_weighted_avg_shift(self.separator_other, mix, out_length, overlap=0.75, shifts=4)

        with torch.no_grad():
            out_length = 1024 * 255
            estimate_vocal = sep_weighted_avg_shift(self.separator_vocal, mix, out_length, overlap=0.75, shifts=4)

        # ------------------------------
        estimate_other[:, -1, ...] = estimate_vocal
        estimate_other[:, 1, ...] = estimate_bass
        estimates = estimate_other

        estimates = estimates.squeeze(0)
        separated_music_arrays = {}
        output_sample_rates = {}
        for instrument in self.instruments:
            idx = self.sources.index(instrument)
            separated_music_arrays[instrument] = torch.squeeze(estimates[idx]).detach().cpu().numpy().T
            output_sample_rates[instrument] = sample_rate

        return separated_music_arrays, output_sample_rates


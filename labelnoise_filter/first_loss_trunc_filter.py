
from pathlib import Path
import time
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio
import torch
import sys


class LTDemucsWrapper:
    """
    Demucs model for music demixing.
    """
    def __init__(self):
    
        # 20230414 pretrain 200 epoch lr=1e-4 trunc=0.3
        self.separator = pretrained.get_model('164037a4', repo=Path('checkpoints'))


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
        # create input for Demucs model
        mix = torch.from_numpy(np.asarray(mixed_sound_array.T, np.float32))
        # convert audio to GPU
        mix = mix.to(self.device)
        mix_channels = mix.shape[0]
        mix = convert_audio(mix, sample_rate, self.separator.samplerate, self.separator.audio_channels)

        with torch.no_grad():
            estimates = apply_model(self.separator, mix[None], shifts=1, overlap=0.25, progress=False)[0]

        sr = self.separator.samplerate
        estimates = convert_audio(estimates, self.separator.samplerate, sample_rate, mix_channels)

        separated_music_arrays = {}
        output_sample_rates = {}
        for instrument in self.instruments:
            idx = self.separator.sources.index(instrument)
            separated_music_arrays[instrument] = torch.squeeze(estimates[idx]).detach().cpu().numpy().T
            output_sample_rates[instrument] = sample_rate

        return separated_music_arrays, output_sample_rates

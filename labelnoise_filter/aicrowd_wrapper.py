## DO NOT CHANGE THIS FILE

import os
from os.path import join as ospj
import numpy as np
import soundfile
import torch
from user_config import MySeparationModel
import tempfile
import wave

def change_pitch_tempo(file_path, start_idx=0, seg_len=50000000,
                       delta_pitch = 6, delta_tempo = 200): # raise 6 semi-tone, speed up 2x.
    with wave.open(file_path, 'rb') as wav_file:
        # Get basic info
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()

        # Seek to the start position in the file
        wav_file.setpos(start_idx)

        # Read the audio data for the specific length by using the pointer we get above
        raw_data = wav_file.readframes(seg_len)

        infile = tempfile.NamedTemporaryFile(suffix=".wav")
        outfile = tempfile.NamedTemporaryFile(suffix=".wav")

        with wave.open(infile.name, 'wb') as temp_wav:
            # Set the parameters for the new temp file
            temp_wav.setnchannels(num_channels)
            temp_wav.setsampwidth(sample_width)
            temp_wav.setframerate(sample_rate)
            temp_wav.writeframes(raw_data)

        
        
        delta_tempo = delta_tempo / 100.0

        # Set Command
        command = [
            "../rubberband-r3",
            f"--pitch {delta_pitch}",
            f"--tempo {delta_tempo:.6f}",
            f"--fast",
            f"--quiet",
            infile.name,
            outfile.name,
            "> /dev/null 2>&1"
        ]

        command = ' '.join(command)
        os.system(command)

        new_wav, sr = soundfile.read(outfile.name)
        new_wav = new_wav[..., :]

        return new_wav, sr


class AIcrowdWrapper:

    def __init__(self,
                 dataset_dir='./public_dataset/',
                 predictions_dir='./evaluator_outputs/'):
                 
        self.model = MySeparationModel()
        self.instruments = ['bass', 'drums', 'other', 'vocals']
        shared_dir = os.getenv("AICROWD_PUBLIC_SHARED_DIR", None)
        if shared_dir is not None:
            self.predictions_dir = os.path.join(shared_dir, 'predictions')
        else:
            self.predictions_dir = predictions_dir
        assert os.path.exists(self.predictions_dir), f'{self.predictions_dir} - No such directory'
        self.dataset_dir = os.getenv("AICROWD_DATASET_DIR", dataset_dir)
        assert os.path.exists(self.dataset_dir), f'{self.dataset_dir} - No such directory'

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs """
        raise NameError(msg)

    def check_output(self, separated_music_arrays, output_sample_rates):
        assert set(self.instruments) == set(separated_music_arrays.keys()), "All instrument not present"
    
    def save_prediction(self, prediction_path, separated_music_arrays, output_sample_rates):
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
            
        for instrument in self.instruments:
            full_path = os.path.join(prediction_path, f'{instrument}.wav')
            soundfile.write(full_path, 
                            data=separated_music_arrays[instrument],
                            samplerate=output_sample_rates[instrument])
        

    
    def separate_music_file(self, foldername, ins, key, speed):
        # ins = 'bass' # ['bass', 'drums', 'other', 'vocals']
        full_path = os.path.join(self.dataset_dir, foldername, f'{ins}.wav')
        music_array, samplerate = soundfile.read(full_path)
        def is_silent(audio_tensor, threshold_db=-80):
            avg_amplitude = torch.mean(torch.abs(torch.from_numpy(audio_tensor)))
            volume_db = 20 * torch.log10(avg_amplitude)
            return volume_db < threshold_db
        
        if is_silent(music_array):
            # print(foldername, ins, 'silent!')
            separated_music_arrays = dict()
            output_sample_rates = dict()
            for inst in self.instruments:
                separated_music_arrays[inst] = music_array
                output_sample_rates[inst] = samplerate
        else:
            music_array, samplerate = change_pitch_tempo(full_path, delta_pitch = key, delta_tempo = speed)
            separated_music_arrays, output_sample_rates = self.model.separate_music_file(music_array, samplerate)
        
        
        self.check_output(separated_music_arrays, output_sample_rates)

        prediction_path = ospj(ospj(self.predictions_dir, ins), foldername)
        self.save_prediction(prediction_path, separated_music_arrays, output_sample_rates)

        return True

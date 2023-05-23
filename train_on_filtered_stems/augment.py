import os
import random
import tempfile
import wave

import torch
import torchaudio
from torch import nn


class FlipChannels(nn.Module):
    """
    Flip left-right channels, borrowed from Demucs.
    """

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        left = torch.randint(2, (batch, sources, 1, 1), device=wav.device)
        left = left.expand(-1, -1, -1, time)
        right = 1 - left
        wav = torch.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        return wav


class FlipSign(nn.Module):
    """
    Random sign flip, borrowed from Demucs.
    """

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        signs = torch.randint(2, (batch, sources, 1, 1), device=wav.device, dtype=torch.float32)
        wav = wav * (2 * signs - 1)
        return wav


class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples, borrowed from Demucs.
    """

    def __init__(self, shift=44100):
        super().__init__()
        self.shift = shift

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        length = time - self.shift
        offsets = torch.randint(self.shift, [batch, sources, 1, 1], device=wav.device)
        offsets = offsets.expand(-1, sources, channels, -1)
        indexes = torch.arange(length, device=wav.device)
        wav = wav.gather(3, indexes + offsets)
        return wav


class Scale(nn.Module):
    def __init__(self, proba=1., low=0.25, high=1.25):
        super().__init__()
        self.proba = proba
        self.low = low
        self.high = high

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device
        if random.random() < self.proba:
            scales = torch.empty(batch, streams, 1, 1, device=device).uniform_(self.low, self.high)
            wav *= scales
        return wav


def change_pitch_tempo(file_path, start_idx, seg_len,
                       proba=0.2, max_pitch=6, max_tempo=12, tempo_std=5):
    """
    We choose RubberBand command tool to repitch or change the tempo of audio file.
    See https://breakfastquay.com/rubberband/ for more details.

    Step 1: Read the raw audio clip
    Step 2: Save and create a new temp WAV file
    Step 3: Use RubberBand to do pitch/tempo change
    Step 4: Return the new WAV tensor
    """

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

        out_length = int((1 - 0.01 * max_tempo) * seg_len)

        if random.random() < proba:
            delta_pitch = random.randint(-max_pitch, max_pitch)
            delta_tempo = random.gauss(0, tempo_std)
            delta_tempo = max(min(max_tempo, delta_tempo), -max_tempo)
            delta_tempo = (1 + delta_tempo / 100.0)

            # Set Command
            command = [
                "rubberband-r3",
                f"--pitch {delta_pitch}",
                f"--tempo {delta_tempo:.6f}",
                # f"--pitch-hq",
                # f"--fine",
                f"--fast",
                f"--quiet",
                infile.name,
                outfile.name,
                "> /dev/null 2>&1"
            ]

            command = ' '.join(command)
            # print(command)
            os.system(command)

            new_wav, _ = torchaudio.load(outfile.name)
            new_wav = new_wav[..., :out_length]

            return new_wav

        else:
            new_wav, _ = torchaudio.load(infile.name)
            new_wav = new_wav[..., :out_length]
            return new_wav
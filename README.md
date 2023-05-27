# ccom_mdx2023

This branch will show how we address the `Bleeding` task, which include a noval Multi-spectrogram Unet model and some interesting observations.

## MultiSUnet (Multi-spectrogram Unet)

Different window lengths and hop lengths in the Short-Time Fourier Transform (STFT) process have a significant impact on the results of source separation, as observed in studies like MDX-Net. 

Shorter window lengths excel in capturing rapid changes in the signal with improved time resolution. 
On the other hand, longer window lengths enhance frequency resolution by narrowing the main lobe in the frequency domain.
There is a trade-off between time and frequency resolution in the STFT, we assume by combining the different STFT spectrogram may help the model to capture both the time and frequency features.

### simple multi spectrogram code example
```python3
z_lst = []
for n_fft, window in zip(self.n_ffts, self.windows):
    z = self.stft(t_input, n_fft, self.hop_length, window)
    z = z[:, :, :2048, :256]
    z = self.magnitude(z)
    z_lst.append(z)

x = torch.cat(z_lst, dim=1)
```

## General Training process

1. Train vocal for 360 epochs --> (`n_ffts=[4096, 6144, 8192], hop_length=1024`)

2. Train bass for 360 epochs --> (`n_ffts=[16384], hop_length=1024`)

3. Train four sources simontenesly for 360 epoch and only take drum/other results during inference. --> (`n_ffts=[4096, 6144, 8192], hop_length=1024`)


## Grade

- Final Scores in Bleeding Leaderboard (Rank 4th):

  | Mean SDR | SDR Bass | SDR Drums | SDR Other | SDR Vocals |
  | -------- | -------- | --------- |-----------|----------  |
  | 6.203    | 6.337    | 6.323     | 4.284     | 7.867     |

import torch
import torch.nn as nn
import torch.nn.functional as F


class Convs(nn.Module):
    def __init__(self, channels, freq_dims, time_dims, kernel_size, padding, stride):
        super(Convs, self).__init__()

        self.conv = nn.ModuleList()
        for i in range(3):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                          stride=stride, padding=padding),
                nn.BatchNorm2d(channels, momentum=0.1),
                nn.ReLU()))

        self.linear_f = nn.Sequential(
                    nn.Linear(freq_dims, freq_dims // 8, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Linear(freq_dims // 8, freq_dims, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU()
                )

        self.linear_t = nn.Sequential(
                    nn.Linear(time_dims, time_dims // 2, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Linear(time_dims // 2, time_dims, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU()
                )

    def forward(self, x):
        skip = x
        for conv in self.conv:
            x = conv(x)
        x = x + self.linear_f(x)

        x = x.transpose(-1, -2)
        x = x + self.linear_t(x)
        x = x.transpose(-1, -2)

        x = x + skip
        return x


class MultiSUnet_v2(nn.Module):
    def __init__(self,
                 num_sources=1, depth=5,
                 n_ffts=[4096, 6144, 8192], hop_length=1024,
                 ):
        super(MultiSUnet_v2, self).__init__()
        self.num_sources = num_sources
        self.depth = depth

        self.n_ffts = n_ffts
        self.hop_length = hop_length
        self.windows = nn.ParameterList(
            [nn.Parameter(torch.hann_window(window_length=n_fft, periodic=True), requires_grad=False) for n_fft in
             self.n_ffts])

        channels = 32
        growth = 32
        freq_dims = 2048
        time_dims = 256

        self.s_pre_conv = nn.Sequential(nn.Conv2d(in_channels=len(self.n_ffts) * 4, out_channels=channels, kernel_size=(1, 1)),
                                        nn.BatchNorm2d(channels),
                                        nn.GELU())
        self.final_conv = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=4 * self.num_sources, kernel_size=(1, 1)))

        encoders = []
        decoders = []
        down = []
        up = []

        for i in range(self.depth):
            encoders.append(Convs(channels=channels, freq_dims=freq_dims, time_dims=time_dims, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

            down.append(nn.Sequential(
                        nn.Conv2d(in_channels=channels, out_channels=channels + growth, kernel_size=(2, 2), stride=(2, 2)),
                        nn.BatchNorm2d(channels + growth),
                        nn.ReLU()))

            decoders.append(Convs(channels=channels, freq_dims=freq_dims, time_dims=time_dims, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

            up.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=channels + growth, out_channels=channels, kernel_size=(2, 2), stride=(2, 2)),
                nn.BatchNorm2d(channels),
                nn.ReLU()))

            channels += growth
            freq_dims //= 2
            time_dims //= 2

        decoders = decoders[::-1]
        up = up[::-1]

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.down = nn.ModuleList(down)
        self.up = nn.ModuleList(up)
        self.bottle = Convs(channels=channels, freq_dims=freq_dims, time_dims=time_dims, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, t_input):
        # -------------- prepare input ----------------
        z_lst = []
        for n_fft, window in zip(self.n_ffts, self.windows):
            z = self.stft(t_input, n_fft, self.hop_length, window)
            z = z[:, :, :2048, :256]
            z = self.magnitude(z)
            z_lst.append(z)

        x = torch.cat(z_lst, dim=1)
        del z_lst

        # ---------------------------------------------
        x = self.s_pre_conv(x)
        x = x.transpose(-1, -2)

        s_res = []
        for encoder, down in zip(self.encoders, self.down):
            x = encoder(x)
            s_res.append(x)
            x = down(x)

        # ------ middle ---------
        x = self.bottle(x)

        for decoder, up in zip(self.decoders, self.up):
            skip = s_res.pop()
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.final_conv(x)
        x = x.transpose(-1, -2)

        x = F.pad(x, pad=(0, 0, 0, 1))
        bs, ch, freq, T = x.shape
        x = x.view(bs, self.num_sources, 2, 2, freq, T).permute(0, 1, 2, 4, 5, 3)
        x = torch.view_as_complex(x.contiguous())
        x = self.istft(x, self.n_ffts[0], self.hop_length, self.windows[0])

        return x

    def stft(self, x, n_fft, hop_length, window):
        bs, ch, seg = x.shape
        x = x.reshape(bs * ch, seg)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length,
                       window=window,
                       center=True,
                       return_complex=True, )
        num_freq = x.shape[1]
        num_frames = x.shape[2]
        x = x.reshape(bs, ch, num_freq, num_frames)

        return x

    def istft(self, x, n_fft, hop_length, window):
        bs, sources, ch, feq, T = x.shape
        x = x.reshape(bs * sources * ch, feq, T)
        x = torch.istft(x, n_fft=n_fft, hop_length=hop_length,
                        window=window,
                        center=True)
        seg = x.shape[-1]
        x = x.reshape(bs, sources, ch, seg)
        return x

    def magnitude(self, z):
        bs, ch, freq, T = z.shape
        mag = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
        mag = mag.reshape(bs, ch * 2, freq, T)
        return mag


class MultiSUnet_v2_bass(nn.Module):
    def __init__(self,
                 num_sources=1, depth=5,
                 n_ffts=[16384], hop_length=1024,
                 freq_dims=2048,
                 ):
        super(MultiSUnet_v2_bass, self).__init__()
        self.num_sources = num_sources
        self.depth = depth
        self.freq_dims = freq_dims

        self.n_ffts = n_ffts
        self.hop_length = hop_length
        self.windows = nn.ParameterList(
            [nn.Parameter(torch.hann_window(window_length=n_fft, periodic=True), requires_grad=False) for n_fft in
             self.n_ffts])

        channels = 32
        growth = 32

        time_dims = 256

        self.s_pre_conv = nn.Sequential(nn.Conv2d(in_channels=len(self.n_ffts) * 4, out_channels=channels, kernel_size=(1, 1)),
                                        nn.BatchNorm2d(channels),
                                        nn.GELU())
        self.final_conv = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=4 * self.num_sources, kernel_size=(1, 1)))

        encoders = []
        decoders = []
        down = []
        up = []

        for i in range(self.depth):
            encoders.append(Convs(channels=channels, freq_dims=freq_dims, time_dims=time_dims, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

            down.append(nn.Sequential(
                        nn.Conv2d(in_channels=channels, out_channels=channels + growth, kernel_size=(2, 2), stride=(2, 2)),
                        nn.BatchNorm2d(channels + growth),
                        nn.ReLU()))

            decoders.append(Convs(channels=channels, freq_dims=freq_dims, time_dims=time_dims, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

            up.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=channels + growth, out_channels=channels, kernel_size=(2, 2), stride=(2, 2)),
                nn.BatchNorm2d(channels),
                nn.ReLU()))

            channels += growth
            freq_dims //= 2
            time_dims //= 2

        decoders = decoders[::-1]
        up = up[::-1]

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.down = nn.ModuleList(down)
        self.up = nn.ModuleList(up)
        self.bottle = Convs(channels=channels, freq_dims=freq_dims, time_dims=time_dims, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, t_input):
        # -------------- prepare input ----------------
        z_lst = []
        for n_fft, window in zip(self.n_ffts, self.windows):
            z = self.stft(t_input, n_fft, self.hop_length, window)
            z = z[:, :, :self.freq_dims, :256]
            z = self.magnitude(z)
            z_lst.append(z)

        x = torch.cat(z_lst, dim=1)
        del z_lst

        # ---------------------------------------------
        x = self.s_pre_conv(x)
        x = x.transpose(-1, -2)

        s_res = []
        for encoder, down in zip(self.encoders, self.down):
            x = encoder(x)
            s_res.append(x)
            x = down(x)

        # ------ middle ---------
        x = self.bottle(x)

        for decoder, up in zip(self.decoders, self.up):
            skip = s_res.pop()
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.final_conv(x)
        x = x.transpose(-1, -2)

        x = F.pad(x, pad=(0, 0, 0, self.n_ffts[0] // 2 + 1 - self.freq_dims))
        bs, ch, freq, T = x.shape
        x = x.view(bs, self.num_sources, 2, 2, freq, T).permute(0, 1, 2, 4, 5, 3)
        x = torch.view_as_complex(x.contiguous())
        x = self.istft(x, self.n_ffts[0], self.hop_length, self.windows[0])

        return x

    def stft(self, x, n_fft, hop_length, window):
        bs, ch, seg = x.shape
        x = x.reshape(bs * ch, seg)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length,
                       window=window,
                       center=True,
                       return_complex=True, )
        num_freq = x.shape[1]
        num_frames = x.shape[2]
        x = x.reshape(bs, ch, num_freq, num_frames)

        return x

    def istft(self, x, n_fft, hop_length, window):
        bs, sources, ch, feq, T = x.shape
        x = x.reshape(bs * sources * ch, feq, T)
        x = torch.istft(x, n_fft=n_fft, hop_length=hop_length,
                        window=window,
                        center=True)
        seg = x.shape[-1]
        x = x.reshape(bs, sources, ch, seg)
        return x

    def magnitude(self, z):
        bs, ch, freq, T = z.shape
        mag = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
        mag = mag.reshape(bs, ch * 2, freq, T)
        return mag


# if __name__ == '__main__':
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = "cpu"
#     # # print(device)
#     t_input = torch.rand(1, 2, 255 * 1024).to(device)
#     model = MultiSUnet_v2_bass(num_sources=1, n_ffts=[4096, 6144, 8192, 16384])
#     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"The number of parameters in the model is: {num_params}")
#     model.to(device)
#     out = model(t_input)
#     print(out)

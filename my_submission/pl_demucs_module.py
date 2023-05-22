import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import sys
sys.path.append('my_submission/')
import random
from demucs.train import get_model
from pathlib import Path

class LitModuleWrapper(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.model= get_model(args)
        self.model_bag = torch.nn.ModuleDict()
        for source in args.dset.sources:
            self.model_bag[source] = get_model(args)

    def forward(self, mix): # deprecated
        # x = self.model(mix)
        # return x
        res = []
        for source in self.args.dset.sources:
            res.append(self.model_bag[source](mix))
        return res
    

def sep_weighted_avg(model, mix, seg_len, overlap=0.25, transition_power=1.0):
    bs, ch, T = mix.shape

    stride = int((1 - overlap) * seg_len)
    num_missing = seg_len - (T % stride)
    mix = F.pad(mix, (0, num_missing))

    out = torch.zeros(bs, 4, ch, mix.shape[-1], device=mix.device)
    sum_weight = torch.zeros(mix.shape[-1], device=mix.device)
    weight = torch.cat([torch.arange(1, seg_len // 2 + 1, device=mix.device),
                        torch.arange(seg_len - seg_len // 2, 0, -1, device=mix.device)])
    weight = (weight / weight.max())**transition_power

    for start_idx in range(0, mix.shape[-1] - seg_len + 1, stride):
        chunk = mix[..., start_idx: start_idx + seg_len]
        chunk_out = model(chunk)
        out[..., start_idx:start_idx + seg_len] += (weight * chunk_out)
        sum_weight[start_idx:start_idx + seg_len] += weight

    out /= sum_weight
    out = out[..., :-num_missing]

    return out

def sep_weighted_avg_shift(model, mix, seg_len, overlap=0.75, transition_power=1.0, shifts=3):
    max_shift = int(0.5 * 44100)
    out = 0
    for _ in range(shifts):
        shift = random.randint(0, max_shift)
        shifted_mix = F.pad(mix, (shift, 0))  # pad at left
        shifted_out = sep_weighted_avg(model, shifted_mix, seg_len, overlap=overlap, transition_power=transition_power)
        out += shifted_out[..., shift:]

    out /= shifts
    return out
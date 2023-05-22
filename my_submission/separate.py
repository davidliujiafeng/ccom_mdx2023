import torch
import torch.nn.functional as F
import random


def separate_simple(model, mix, seg_len):
    """
    This is a naive method to separate audio without overlap
    """
    num_missing = seg_len - (mix.shape[-1] % seg_len)
    mix = F.pad(mix, (0, num_missing))
    chunks = torch.split(mix, seg_len, dim=-1)
    estimates = []
    for chunk in chunks:
        estimate_ = model(chunk)
        estimates.append(estimate_)

    estimate = torch.cat(estimates, dim=-1)
    estimate = estimate[..., :-num_missing]

    return estimate


def separate_simple_v2(model, mix, seg_len):
    """
    This is a naive method to separate audio without overlap
    """
    num_missing = seg_len - (mix.shape[-1] % seg_len)
    mix = F.pad(mix, (0, num_missing))
    chunks = torch.split(mix, seg_len, dim=-1)
    estimates = []
    for chunk in chunks:
        _, estimate_ = model(chunk)
        estimates.append(estimate_)

    estimate = torch.cat(estimates, dim=-1)
    estimate = estimate[..., :-num_missing]

    return estimate


def sep_weighted_avg(model, mix, seg_len, overlap=0.25, transition_power=1.0):
    bs, ch, T = mix.shape

    stride = int((1 - overlap) * seg_len)
    num_missing = seg_len - (T % stride)
    mix = F.pad(mix, (0, num_missing))

    out = torch.zeros(bs, model.num_sources, ch, mix.shape[-1], device=mix.device)
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


def sep_weighted_avg_shift(model, mix, seg_len, overlap=0.25, transition_power=1.0, shifts=1):
    max_shift = int(0.5 * 44100)
    out = 0
    for _ in range(shifts):
        shift = random.randint(0, max_shift)
        shifted_mix = F.pad(mix, (shift, 0))  # pad at left
        shifted_out = sep_weighted_avg(model, shifted_mix, seg_len, overlap=overlap, transition_power=transition_power)
        out += shifted_out[..., shift:]

    out /= shifts
    return out


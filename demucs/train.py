#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main training script entry point"""

import logging
import os
from pathlib import Path
import sys


from omegaconf import OmegaConf
import torch
from torch import nn
from torch.utils.data import ConcatDataset

from .demucs import Demucs
from .hdemucs import HDemucs
from .htdemucs import HTDemucs
from .states import capture_init
from .utils import random_subset

logger = logging.getLogger(__name__)


class TorchHDemucsWrapper(nn.Module):
    """Wrapper around torchaudio HDemucs implementation to provide the proper metadata
    for model evaluation.
    See https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html"""

    @capture_init
    def __init__(self,  **kwargs):
        super().__init__()
        try:
            from torchaudio.models import HDemucs as TorchHDemucs
        except ImportError:
            raise ImportError("Please upgrade torchaudio for using its implementation of HDemucs")
        self.samplerate = kwargs.pop('samplerate')
        self.segment = kwargs.pop('segment')
        self.sources = kwargs['sources']
        self.torch_hdemucs = TorchHDemucs(**kwargs)

    def forward(self, mix):
        return self.torch_hdemucs.forward(mix)


def get_model(args):
    extra = {
        'sources': list(args.dset.sources),
        'audio_channels': args.dset.channels,
        'samplerate': args.dset.samplerate,
        'segment': args.model_segment or 4 * args.dset.segment,
    }
    klass = {
        'demucs': Demucs,
        'hdemucs': HDemucs,
        'htdemucs': HTDemucs,
        'torch_hdemucs': TorchHDemucsWrapper,
    }[args.model]
    kw = OmegaConf.to_container(getattr(args, args.model), resolve=True)
    model = klass(**extra, **kw)
    return model


def get_optimizer(model, args):
    seen_params = set()
    other_params = []
    groups = []
    for n, module in model.named_modules():
        if hasattr(module, "make_optim_group"):
            group = module.make_optim_group()
            params = set(group["params"])
            assert params.isdisjoint(seen_params)
            seen_params |= set(params)
            groups.append(group)
    for param in model.parameters():
        if param not in seen_params:
            other_params.append(param)
    groups.insert(0, {"params": other_params})
    parameters = groups
    if args.optim.optim == "adam":
        return torch.optim.Adam(
            parameters,
            lr=args.optim.lr,
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    elif args.optim.optim == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=args.optim.lr,
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    else:
        raise ValueError("Invalid optimizer %s", args.optim.optimizer)




import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import augment
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from demucs.train import get_model, get_optimizer

class LitModuleWrapper(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = get_model(args)

        # data augment
        augments = [
            augment.Shift(),
            augment.FlipChannels(),
            augment.FlipSign(),
            augment.Scale()
        ]

        self.augment = torch.nn.Sequential(*augments)

    def forward(self, mix):
        x = self.model(mix)
        return x

    def training_step(self, batch, batch_idx):
        sources = self.augment(batch)
        mix = sources.sum(dim=1)
        estimate = self(mix)

        dims = tuple(range(2, sources.dim()))
        loss = F.l1_loss(estimate, sources, reduction='none')
        loss = loss.mean(dims).mean(0)
        weights = torch.tensor(self.args.weights, device=sources.device)
        loss = (loss * weights).sum() / weights.sum()

        return loss

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model, self.args)
        return optimizer
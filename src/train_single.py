import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import augment
from data_module import WaveBleedingDataModule
from models.MultiSUnet_v2 import MultiSUnet_v2


class TrainModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.model = MultiSUnet_v2()

        # data augment
        augments = [
            augment.Shift(),
            augment.FlipChannels(),
            augment.FlipSign(),
            augment.Scale(),
            augment.Remix(),
        ]
        self.source_name = ["drums", "bass", "other", "vocals"]
        self.train_idx = -1
        self.augment = torch.nn.Sequential(*augments)

    def forward(self, mix):
        x = self.model(mix)
        return x

    def training_step(self, batch, batch_idx):

        sources = self.augment(batch)
        mix = sources.sum(dim=1)
        estimate = self(mix)
        # print(estimate.shape)

        # time domain loss
        source = sources[:, self.train_idx, ...].unsqueeze(1)
        loss_t = F.l1_loss(estimate, source)
        # print(estimate.shape, source.shape)

        total_loss = loss_t

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.args.optim.lr,
                                      )

        return [optimizer]


if __name__ == "__main__":
    from omegaconf import OmegaConf

    my_args = OmegaConf.load('config/bleeding.yaml')

    # Prepare for DataModule.
    dm = WaveBleedingDataModule(my_args)
    dm.setup()

    # Train the model
    model = TrainModule(my_args)
    trainer = pl.Trainer(
        accelerator="auto", devices=8,
        strategy="ddp_find_unused_parameters_false",
        accumulate_grad_batches=16,
        max_epochs=3000,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=dm)


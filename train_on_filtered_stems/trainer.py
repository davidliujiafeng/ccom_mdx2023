import pytorch_lightning as pl
from omegaconf import OmegaConf

from datasets import WaveLabelDataModule
from module import LitModuleWrapper


def main():

    train_args = OmegaConf.load('../config/labelnoise.yaml')
    # Prepare for DataModule.
    dm = WaveLabelDataModule(train_args)
    dm.setup()

    # Train the model
    model = LitModuleWrapper(train_args)

    trainer = pl.Trainer(
        accelerator="auto", devices=8, strategy="ddp_find_unused_parameters_false",
        accumulate_grad_batches=16,
        max_epochs=3000,
        num_sanity_val_steps=0
    )
    
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
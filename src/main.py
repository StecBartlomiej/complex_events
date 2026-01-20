from pytorch_lightning import Trainer, seed_everything
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(version_base=None, config_path="../config", config_name='config')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed, workers=True)

    logger = instantiate(cfg.logger)

    checkpoint = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{val_acc:.3f}",
    )

    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        default_root_dir='logs',
        logger=logger,
        callbacks=[checkpoint],
        # gradient_clip_val=0.05,
        # gradient_clip_algorithm='norm'
    )

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)

    trainer.fit(model, datamodule=datamodule)

    trainer.test(ckpt_path='best', datamodule=datamodule)

if __name__ == "__main__":
    main()

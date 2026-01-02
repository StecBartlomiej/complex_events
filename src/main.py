from pytorch_lightning import Trainer
from models.cifar_paper_model import ComplexCifarPaper
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(version_base=None, config_path="../config", config_name='config')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

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
        callbacks=[checkpoint]
    )

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)

    trainer.fit(model, datamodule=datamodule)
    trainer.validate(datamodule=datamodule)
    trainer.test(datamodule=datamodule)

if __name__ == "__main__":
    main()

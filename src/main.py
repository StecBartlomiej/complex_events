from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor
from models.cifar_model import ComplexCifar
from models.cifar_paper_model import ComplexCifarPaper
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import numpy as np


@hydra.main(version_base=None, config_path="../config", config_name='config')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger = instantiate(cfg.logger)

    trainer = Trainer(
        max_epochs=6,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        default_root_dir='logs',
        logger=logger
    )

    with trainer.init_module():
        datamodule = instantiate(cfg.datamodule)
        model = ComplexCifarPaper(in_ch=1, lr=cfg.lr)


    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

if __name__ == "__main__":
    main()

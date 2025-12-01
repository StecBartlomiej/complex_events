from pytorch_lightning import Trainer
from models.mnist_model import ComplexMNISTLightning
from models.cifar_model import ComplexCifar
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch


@hydra.main(version_base=None, config_path="../config", config_name='config')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    model = ComplexCifar(in_ch=1, lr=0.001)

    datamodule = instantiate(cfg.datamodule)
    logger = instantiate(cfg.logger)


    model = torch.compile(model)

    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        default_root_dir='logs'
#        logger=logger
    )

    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()

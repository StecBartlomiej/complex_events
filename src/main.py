from pytorch_lightning import Trainer
from data.mnist_datamodule import MNISTDataModule
from models.mnist_model import ComplexMNISTLightning


def main():
    model = ComplexMNISTLightning(lr=1e-3)
    datamodule = MNISTDataModule(batch_size=64)

    trainer = Trainer(
        max_epochs=3,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        default_root_dir='logs'
    )

    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()

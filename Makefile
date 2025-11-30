
.PHONY: train_mnist

train_mnist:
	python3 src/main.py datamodule=mnist

train_cifar10:
	python3 src/main.py datamodule=cifar

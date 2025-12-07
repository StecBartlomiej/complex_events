
.PHONY: train_mnist

fft_cifar10:
	python3 src/utils/precompute_fft.py

raw_bins_cifar10:
	python3 src/utils/precompute_raw_bins.py

train_mnist:
	python3 src/main.py datamodule=mnist

train_cifar10:
	python3 src/main.py datamodule=cifar


.PHONY: train_mnist fft_cifar10 raw_bins_cifar10 train_cifar_paper train_cifar_small
 
# UTILS
fft_cifar10:
	python3 src/utils/precompute_fft.py

raw_bins_cifar10:
	python3 src/utils/precompute_raw_bins.py

# MNIST
train_mnist:
	python3 src/main.py datamodule=mnist

# CIFAR
train_cifar_paper:
	python3 src/main.py datamodule=cifar model=cifar_paper

train_cifar_small:
	python3 src/main.py datamodule=cifar model=cifar_small

import os
from pathlib import Path
import numpy as np
import tonic
from tqdm import tqdm
import multiprocessing as mp


class CIFAR10DVSPreprocessor:
    def __init__(self,
                 dataset_path="./data",
                 output_root="./data/cifar10dvs_fft_bins",
                 time_step=100_000,
                 sensor_size=(128, 128),
                 num_workers=None):

        self.data_root = Path(dataset_path).parent
        self.output_root = Path(output_root)
        self.time_step = time_step
        self.sensor_size = sensor_size
        self.num_workers = num_workers 
        self.dataset_path = Path(dataset_path)

    def _ensure_dataset(self):
        if not self.dataset_path.exists() or not any(self.dataset_path.iterdir()):
            print("Dataset not found, downloading CIFAR10-DVS...")
            tonic.datasets.CIFAR10DVS(save_to=str(self.data_root))

    def _scan_dataset(self):
        file_names = [p for p in self.dataset_path.iterdir()]
        classes = sorted([p for p in file_names if p.is_dir()])
        class_to_idx = {cls.name: i for i, cls in enumerate(classes)}

        workload = []
        for cls_folder in classes:
            label = class_to_idx[cls_folder.name]
            for f in cls_folder.iterdir():
                if f.name.endswith(".aedat4"):
                    workload.append((f, label))
        return workload, class_to_idx

    def _process_single(self, args):
        file_path, label = args
        file_path: Path

        class_folder = self.output_root / str(label)
        class_folder.mkdir(parents=True, exist_ok=True)

        events = tonic.io.read_aedat4(str(file_path))
        t = np.array([e[0] for e in events])
        x = np.array([e[1] for e in events])
        y = np.array([e[2] for e in events])
        p = np.array([float(e[3]) for e in events])

        t_min, t_max = t.min(), t.max()
        n_bins = int(np.ceil((t_max - t_min) / self.time_step))

        voxel = np.zeros((n_bins, self.sensor_size[0], self.sensor_size[1]), dtype=np.float32)

        bin_idx = ((t - t_min) / self.time_step).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        for b, xi, yi, pi in zip(bin_idx, x, y, p):
            voxel[b, xi, yi] += pi

        for bin_id in range(n_bins):
            fft_data = np.fft.fft2(voxel[bin_id])

            out_path = class_folder / f"{file_path.stem}_bin{bin_id}.npz"
            np.savez_compressed(out_path, fft=fft_data, label=label)
        return n_bins

    def run(self):
        self._ensure_dataset()

        workload, class_map = self._scan_dataset()
        print(f"Class mapping: {class_map}")
        print(f"Found {len(workload)} recordings to process.")

        self.output_root.mkdir(parents=True, exist_ok=True)

        print(f"Processing using {self.num_workers} workers...")
        with mp.Pool(self.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(self._process_single, workload),
                          total=len(workload)):
                pass
        print("Preprocessing completed successfully.")


if __name__ == "__main__":
    pre = CIFAR10DVSPreprocessor(
        dataset_path="./data/CIFAR10DVS",
        output_root="./data/cifar10dvs_fft_bins",
        time_step=100_000,
        sensor_size=(128, 128),
        num_workers=8
    )
    pre.run()

import os
from pathlib import Path
import numpy as np
import tonic
from tqdm import tqdm
import multiprocessing as mp
import json


class CIFAR10DVSPreprocessor:
    def __init__(self,
                 dataset_path="./data",
                 output_root="./data/cifar10dvs_fft_bins",
                 time_step=100_000,
                 sensor_size=(128, 128),
                 num_workers=None,
                 data_represenation="voxel",
                 pos_polarity=True,
                 neg_polarity=False,  # Jak chcemy mieć dwa kanały dla polarity w voxel grid, zamiast brać tylko pozytywnych
                 max_events_per_px=5
                 ):
        self.data_root = Path(dataset_path).parent
        self.output_root = Path(output_root)
        self.time_step = time_step
        self.sensor_size = sensor_size
        self.num_workers = num_workers 
        self.dataset_path = Path(dataset_path)
        self.data_representation = data_represenation  # "voxel" | "exp_frames"
        self.pos_polarity = pos_polarity
        self.neg_polarity = neg_polarity
        self.max_events_per_px = max_events_per_px

    def to_dict(self):
        return {
            "data_root": str(self.data_root),
            "output_root": str(self.output_root),
            "time_step": self.time_step,
            "sensor_size": self.sensor_size,
            "num_workers": self.num_workers,
            "dataset_path": str(self.dataset_path),
        }

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
    
    def __build_raw_voxel_grid(self, t, x, y, p, n_bins):
        """
        Tworzy voxel grid na podstawie zdarzeń z p==1.
        Parameters
        x, y : ndarray
            Współrzędne pikseli zdarzeń
        p : ndarray
            Polaryzacja zdarzeń (0/1)
        n_bins : int
            Liczba przedziałów czasowych

        Returns
        frames : ndarray of shape (n_bins, H, W)
            Event frame'y
        """
        H, W = self.sensor_size
        voxel = np.zeros((n_bins, H, W), dtype=np.float32)

        bin_idx = ((t - t.min()) / self.time_step).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        for b, xi, yi, pi in zip(bin_idx, x, y, p):
            voxel[b, xi, yi] += pi

        voxel = np.clip(voxel, 0, self.max_events_per_px)
        
        return voxel
    
    def __build_exp_frames(self, t, x, y, p, n_bins, delta_t):
        """
        Tworzy event freme'y za pomocą wykładniczego zanikania (exponential decay).
        
        Parameters
        t, x, y, p : ndarray
            Dane zdarzeń
        n_bins : int
            Ilość binów czasowych
        delta_t : float
            Stała czasowa zanikania wykładniczego

        Returns
        frames : ndarray shape (n_bins, H, W)
            Event frames zbudowane metodą wykładniczą
        """

        H, W = self.sensor_size
        frames = np.zeros((n_bins, H, W), dtype=np.float32) + 128

        t_sec = (t - t.min()) * 1e-6  # us → s

        p[p==0] = -1

        bin_idx = ((t - t.min()) / self.time_step).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        t_max_bins = np.zeros(n_bins, dtype=t.dtype)
        for b in range(n_bins):
            mask = bin_idx == b
            if np.any(mask):
                t_max_bins[b] = t_sec[mask].max()  # lub jak jest pewna chronologiczna kolejność t_sec[mask][-1]

        decay = np.exp(-(t_max_bins[bin_idx] - t_sec) / delta_t)

        np.maximum.at(frames, (bin_idx, x, y), (p * decay + 1.0) * (255.0 / 2.0))

        return frames.astype(int)

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

        voxel = []

        if self.data_representation == "exp_frames":
            voxel.append(self.__build_exp_frames(t, x, y, p, n_bins, delta_t=0.06))
        else:  # self.data_representation == "voxel"
            if self.pos_polarity:
                voxel.append(self.__build_raw_voxel_grid(t, x, y, p, n_bins))
            if self.neg_polarity:
                p = -(p-1)
                voxel.append(self.__build_raw_voxel_grid(t, x, y, p, n_bins))

        voxel = np.array(voxel).transpose((1,0,2,3))  #  pol_channels, n_bins, h, w -> n_bins, pol_channels, h, w
        if voxel.shape[1] == 1:
            voxel = voxel.squeeze(1)  # n_bins, 1, h, w -> n_bins, h, w

        for bin_id in range(n_bins):
            out_path = class_folder / f"{file_path.stem}_{bin_id}_raw.npz"
            np.savez_compressed(out_path, data=voxel[bin_id], label=label)
        return n_bins

    def run(self):
        self._ensure_dataset()

        workload, class_map = self._scan_dataset()
        print(f"Class mapping: {class_map}")
        print(f"Found {len(workload)} recordings to process.")

        self.output_root.mkdir(parents=True, exist_ok=True)

        with open(self.output_root / 'config.json', 'w') as f:
            json.dump(self.to_dict(), f)

        print(f"Processing using {self.num_workers} workers...")
        with mp.Pool(self.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(self._process_single, workload),
                          total=len(workload)):
                pass
        print("Preprocessing completed successfully.")


if __name__ == "__main__":
    pre = CIFAR10DVSPreprocessor(
        dataset_path="./data/CIFAR10DVS",
        output_root="./data/cifar10dvs_raw_bins",
        time_step=40_000,
        sensor_size=(128, 128),
        num_workers=8,
        data_represenation="voxel",
        # pos_polarity=True,
        # neg_polarity=True
    )
    pre.run()

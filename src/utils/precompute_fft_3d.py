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
                 num_workers=None,
                 window_size=50_000, 
                 step_size=10_000,
                 max_per_class = 50):

        self.data_root = Path(dataset_path).parent
        self.output_root = Path(output_root)
        self.time_step = time_step
        self.sensor_size = sensor_size
        self.num_workers = num_workers 
        self.dataset_path = Path(dataset_path)
        self.window_size = window_size
        self.step_size = step_size
        self.max_per_class = max_per_class

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
            files = [f for f in cls_folder.iterdir() if f.name.endswith(".aedat4")]

            if self.max_per_class is not None:
                files = files[:self.max_per_class]  # bierzemy tylko część z klasy

            for f in files:
                workload.append((f, label))
        return workload, class_to_idx
    
    def _accumulate_event_slices(self, t, x, y, p, slice_size=10_000, num_slices=5):

        voxel = np.zeros((num_slices, self.sensor_size[0], self.sensor_size[1]), dtype=np.float32)
        t_start = t.min()

        for i in range(num_slices):
            start = t_start + i * slice_size
            end = start + slice_size
            mask = (t >= start) & (t < end)
            for xi, yi, pi in zip(x[mask], y[mask], p[mask]):
                voxel[i, xi, yi] += pi

        return voxel



    def _process_single(self, args):
        file_path, label = args
        class_folder = self.output_root / str(label)
        class_folder.mkdir(parents=True, exist_ok=True)

        events = tonic.io.read_aedat4(str(file_path))
        t = np.array([e[0] for e in events])
        x = np.array([e[1] for e in events])
        y = np.array([e[2] for e in events])
        p = np.array([float(e[3]) for e in events])

        num_slices = self.window_size // self.step_size

        t_min, t_max = t.min(), t.max()
        n_windows = int(np.ceil((t_max - t_min - self.window_size) / self.step_size)) + 1

        for w in range(n_windows):
            start = t_min + w * self.step_size
            end = start + self.window_size
            mask = (t >= start) & (t < end)

            voxel_window = self._accumulate_event_slices(
                t[mask], x[mask], y[mask], p[mask],
                slice_size=self.step_size,
                num_slices=num_slices
            )

            fft_data = np.fft.fftn(voxel_window)

            out_path = class_folder / f"{file_path.stem}_window{w}.npz"
            np.savez_compressed(out_path, fft=fft_data, label=label)

        return n_windows

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
        output_root="./data/cifar10dvs_fft_bins3d_test",
        sensor_size=(128, 128),
        num_workers=4,
        window_size=50_000,
        step_size=10_000,
        max_per_class=10  
    )
    pre.run()

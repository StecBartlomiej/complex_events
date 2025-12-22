import numpy as np
import cv2
from pathlib import Path

# output_root = Path("./data/cifar10dvs_raw_bins")  # folder z zapisanymi .npz
# output_root = Path("./data/cifar10dvs_pos_raw_bins")  # folder z zapisanymi .npz
output_root = Path("./data/cifar10dvs_all_raw_bins")  # folder z zapisanymi .npz
# output_root = Path("./data/cifar10dvs_exp_frames")  # folder z zapisanymi .npz
label_to_show = "0"  # klasa do wyświetlenia
delay_ms = 200       # opóźnienie między binami w ms

label_folder = output_root / label_to_show
npz_files = sorted(label_folder.glob("*.npz"))

if not npz_files:
    print(f"Nie znaleziono plików dla klasy {label_to_show}")
    exit()

for npz_file in npz_files:
    with np.load(npz_file) as f:
        frame = f["data"].astype(np.uint8)

    if frame.ndim == 2:
        frame = np.expand_dims(frame, axis=0)

    for p_i in range(frame.shape[0]):
        # Normalizacja
        frame[p_i] = (frame[p_i] - frame[p_i].min()) / (frame[p_i].max() - frame[p_i].min()) * 255

    cv2.imshow(f"Class {label_to_show} - pos", frame[0])
    if frame.shape[0] > 1:
        cv2.imshow(f"Class {label_to_show} - neg", frame[1])

        color = np.zeros((frame[0].shape[0], frame[0].shape[1], 3), dtype=np.uint8)
        color[:,:,0] = frame[0]
        color[:,:,2] = frame[1]
        cv2.imshow(f"Class {label_to_show} - all", color)

    key = cv2.waitKey(delay_ms)
    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

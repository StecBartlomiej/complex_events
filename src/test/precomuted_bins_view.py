import numpy as np
import cv2
from pathlib import Path

output_root = Path("./data/cifar10dvs_exp_frames")  # folder z zapisanymi .npz
label_to_show = "0"  # klasa do wyświetlenia
delay_ms = 300       # opóźnienie między binami w ms

label_folder = output_root / label_to_show
npz_files = sorted(label_folder.glob("*.npz"))

if not npz_files:
    print(f"Nie znaleziono plików dla klasy {label_to_show}")
    exit()

for npz_file in npz_files:
    with np.load(npz_file) as f:
        frame = f["data"].astype(np.uint8)

    # skalowanie do 0-255 jeśli trzeba
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)

    # Normalizacja
    # frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255

    # wyświetlenie w oknie
    cv2.imshow(f"Class {label_to_show}", frame)
    
    key = cv2.waitKey(delay_ms)
    # jeśli naciśnięto 'q', wyjdź
    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

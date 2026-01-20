import tonic
from tonic.slicers import SliceByTime
from tonic.sliced_dataset import SlicedDataset
import random


LABEL_TO_NAME = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
}


base_ds = tonic.datasets.CIFAR10DVS("./data")


slicer = SliceByTime(
    time_window=100_000,
    overlap=10_000,
    include_incomplete=False,
)

# =========== Frame Representation ===============

transform = tonic.transforms.ToVoxelGrid(
    sensor_size=base_ds.sensor_size,
    n_time_bins=6
)

# transform = tonic.transforms.Compose([ # type: ignore
#         tonic.transforms.ToTimesurface(
#             sensor_size=base_ds.sensor_size,
#             tau=15_000,
#             dt=10_000,
#         ),
#         ]
#     )



# =========== END ===============


ds = SlicedDataset(base_ds, # type: ignore 
                   slicer=slicer, # type: ignore 
                   # metadata_path='metadata/view_utils'
                   ) 


while True:
    i = random.randint(0, len(ds) - 1)
    events, label = ds[i]

    frames = transform(events)
    print(frames.shape)
    print(LABEL_TO_NAME[label])

    ani = tonic.utils.plot_animation(frames)

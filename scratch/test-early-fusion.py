from mmpose.apis import init_pose_model, inference_top_down_pose_model
from meerkat.training_data import MeerkatDataLoader
import numpy as np


config_file = "early-fusion-config.py"
file_names = [
    "test_archive:color:-1_-1_-1_-1",
    "test_archive:depth:-1_-1_-1_-1",
    "test_archive:ir:-1_-1_-1_-1",
]
file_name = "::".join(file_names)
loader = MeerkatDataLoader()
images = [loader.load_image(f) for f in file_names]
print([i.shape for i in images])

new_image = np.zeros((720, 1280, 5), dtype=images[0].dtype)
new_image[..., :3] = images[0]
new_image[..., 3] = images[1]
new_image[..., 4] = images[2]

model = init_pose_model(
    config_file, "/Users/alex/dev/mmpose-old/scratch/multi-model.pth", device="cpu"
)
result = inference_top_down_pose_model(model, new_image)
print(result)

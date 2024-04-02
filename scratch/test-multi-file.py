from mmpose.datasets.pipelines import LoadMultipleImagesFromMeerkat
from meerkat import MktReader
from meerkat.training_data import MeerkatDataWriter
import matplotlib.pyplot as plt
import numpy as np


color_order = "bgr"
start_indices = 1
mkt_file = "/data/file_20231005_144216.mkt"
reader = MktReader(mkt_file)
color_img, depth_img, ir_img = reader.read_frame(get_params=False)
depth_img = depth_img.astype(float)
min_depth = np.min(depth_img[depth_img != 0])
max_depth = np.max(depth_img)
depth_img = (depth_img - min_depth) / (max_depth - min_depth) * 255
depth_img = np.clip(depth_img, 0, 255).astype(np.uint8)

ir_img = ir_img.astype(float)
min_depth = np.min(ir_img[ir_img != 0])
max_depth = np.max(ir_img)
ir_img = (ir_img - min_depth) / (max_depth - min_depth) * 255
ir_img = np.clip(ir_img, 0, 255).astype(np.uint8)


writer = MeerkatDataWriter("test_archive")
writer.add_image(color_img, "color")
writer.add_image(depth_img, "depth")
writer.add_image(ir_img, "ir")
writer.write()

file_names = [
    "test_archive:depth:-1_-1_-1_-1",
    "test_archive:color:-1_-1_-1_-1",
    "test_archive:ir:-1_-1_-1_-1",
]
file_name = "::".join(file_names)

results = {"image_file": file_name}
loader = LoadMultipleImagesFromMeerkat(
    color_channel_order=color_order, color_start_indices=start_indices
)
image = loader(results)["img"]

plt.figure()
plt.imshow(image[:, :, 0])
plt.figure()
plt.imshow(image[:, :, [3, 2, 1]])
plt.figure()
plt.imshow(image[:, :, 4])
plt.show()

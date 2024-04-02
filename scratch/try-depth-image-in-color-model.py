import json
from meerkat.training_data import MeerkatDataLoader
from mmpose.apis import inference_top_down_pose_model, init_pose_model
import numpy as np
import os
import matplotlib.pyplot as plt


json_file = "test-multi-annotations.json"
archive_dir = "test-multi-archives"
with open(json_file, "r") as f:
    json_data = json.load(f)
print(json_data["images"][40])
image_file_1 = json_data["images"][0]["file_name"]
color_file_name = image_file_1.split("::")[0]
ir_file_name = image_file_1.split("::")[2]
depth_file_name = image_file_1.split("::")[1]
depth_file_name = os.path.join(archive_dir, depth_file_name)
ir_file_name = os.path.join(archive_dir, ir_file_name)
color_file_name = os.path.join(archive_dir, color_file_name)
loader = MeerkatDataLoader()
depth_image = loader.load_image(depth_file_name)
ir_image = loader.load_image(ir_file_name)
color_image = loader.load_image(color_file_name)
print(depth_image.shape)

img = np.zeros((720, 1280, 5), dtype=np.uint8)
img[:, :, :3] = color_image
img[:, :, 3] = depth_image
img[:, :, 4] = ir_image
config_file = "early-fusion-config.py"
checkpoint_file = "early-fusion-stage2-model.pth"
model = init_pose_model(config_file, checkpoint_file, device="cpu")
model2 = init_pose_model(
    "/Users/alex/Downloads/config-mk001.py",
    "/Users/alex/Downloads/epoch_10-3.pth",
    device="cpu",
)
model3 = init_pose_model("multi-test-config.py", "multi-model.pth", device="cpu")
plt.figure()
plt.imshow(color_image[:, :, [2, 1, 0]])
results = inference_top_down_pose_model(model, img)[0][0]["keypoints"]
results2 = inference_top_down_pose_model(model2, img[:, :, :3])[0][0]["keypoints"]
# results3 = inference_top_down_pose_model(model3, img)[0][0]["keypoints"]

print(results)
print(results2)
# print(results3)
plt.plot(results[:, 0], results[:, 1], "ob")
plt.plot(results2[:, 0], results2[:, 1], "or")
# plt.plot(results3[:, 0], results3[:, 1], "og")
plt.show()

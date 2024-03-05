import numpy as np
import torch
from mmpose.apis import init_pose_model, inference_top_down_pose_model
from mmcv.runner.epoch_based_runner import save_checkpoint
import cv2
import matplotlib.pyplot as plt


image_file = "/Users/alex/Downloads/person.jpg"
img_color = cv2.imread(image_file, cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_gray_3c = np.zeros_like(img_color)
for i in range(3):
    img_gray_3c[..., i] = img_gray
img_gray = np.expand_dims(img_gray, 2)

pretrained_model_path = "/Users/alex/Downloads/td_torso_model.pth"
gray_config = "grayscale-model-config.py"
rgb_config = "color-model-config.py"
init_pose_model(gray_config, pretrained_model_path, device="cpu")
model_params = torch.load(pretrained_model_path)
state_dict = model_params["state_dict"]
state_dict["backbone.conv1.weight"] = torch.sum(
    state_dict["backbone.conv1.weight"], dim=1, keepdim=True
)

output_model_path = "/Users/alex/Downloads/td_torso_model_grayscale.pth"
torch.save(state_dict, output_model_path)

model_gray = init_pose_model(gray_config, output_model_path, device="cpu")
model_rgb = init_pose_model(rgb_config, pretrained_model_path, device="cpu")

results_gray = inference_top_down_pose_model(model_gray, img_gray)
results_gray_3c = inference_top_down_pose_model(model_rgb, img_gray_3c)
results_color = inference_top_down_pose_model(model_rgb, img_color)

save_checkpoint(model_rgb, "test.pth")
loaded = torch.load("test.pth")
print(loaded["state_dict"].keys())
exit()
# print(model_rgb.state_dict().keys())
# exit()

print(results_gray[0][0]["keypoints"])
print(results_gray_3c[0][0]["keypoints"])
print(results_color[0][0]["keypoints"])

plt.figure()
plt.subplot(131)
plt.imshow(img_gray, cmap="gray")
plt.plot(
    results_gray[0][0]["keypoints"][:, 0], results_gray[0][0]["keypoints"][:, 1], "or"
)
plt.subplot(132)
plt.imshow(img_gray_3c)
plt.plot(
    results_gray_3c[0][0]["keypoints"][:, 0],
    results_gray_3c[0][0]["keypoints"][:, 1],
    "or",
)
plt.subplot(133)
plt.imshow(img_color)
plt.plot(
    results_color[0][0]["keypoints"][:, 0], results_color[0][0]["keypoints"][:, 1], "or"
)
plt.show()

import numpy as np
import torch
from mmpose.apis import init_pose_model, inference_top_down_pose_model
from mmcv import Config
import cv2
import matplotlib.pyplot as plt

resnet_pretrained = "/Users/alex/Downloads/resnet18-f37072fd.pth"
main_pretrained = "multi-model.pth"
config_file = "early-fusion-config.py"
config = Config.fromfile(config_file)

resnet_state = torch.load(resnet_pretrained, map_location="cpu")
full_model = torch.load(main_pretrained, map_location="cpu")
full_state = full_model["state_dict"]
head_key = "keypoint_head.final_layer.weight"
full_state[head_key] = full_state[head_key][:, :32, :, :]

fusion_stage = config["model"]["fuse_after_stage"]
output_model = f"early-fusion-stage{fusion_stage}-model.pth"

# Remove models.{1+}.transition{fuse_after_stage+...}...
# Remove models.{1+}.stage{fuse_after_stage+1+}...
state_keys = list(full_state.keys())
for key in state_keys:
    if not key.startswith("models"):
        continue
    segments = key.split(".")
    if len(segments) < 3:
        continue
    index = int(segments[1])
    layer_type = segments[2]
    if layer_type.startswith("transition"):
        layer_index = int(layer_type[10:])
        layer_type = "transition"
    elif layer_type.startswith("stage"):
        layer_index = int(layer_type[5:])
        layer_type = "stage"
    else:
        continue
    if index > 0 and layer_type == "stage" and layer_index >= fusion_stage + 1:
        del full_state[key]
    elif index > 0 and layer_type == "transition" and layer_index >= fusion_stage:
        del full_state[key]

# Add in the fusion backbone
for key in resnet_state.keys():
    if key.startswith("fc"):
        continue
    full_state["fusion_backbone." + key] = resnet_state[key]
linear_layer_size = (
    8
    * config["model"]["selector_head_map_size"][0]
    * config["model"]["selector_head_map_size"][1]
)
full_state["fusion_head.2.weight"] = torch.randn(size=(3, linear_layer_size)) * 0.01

full_state["fusion_head.2.bias"] = torch.ones(size=(3,)) * 0.0

torch.save(full_model, output_model)
model = init_pose_model(config_file, output_model, device="cpu")
image_file = "/Users/alex/Downloads/person.jpg"
img_color = cv2.imread(image_file)[:, 50:300, :]
img_multi = np.zeros_like(img_color, shape=(img_color.shape[0], img_color.shape[1], 5))
img_multi[:, :, :3] = img_color
img_multi[:, :, 3] = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_multi[:, :, 4] = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
results = inference_top_down_pose_model(model, img_multi)[0][0]["keypoints"]

plt.figure()
plt.imshow(img_color)
plt.plot(results[:, 0], results[:, 1], "or")
plt.show()

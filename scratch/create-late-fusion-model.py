import numpy as np
import torch
from mmpose.apis import init_pose_model, inference_top_down_pose_model
import cv2

resnet_pretrained = "/Users/alex/Downloads/resnet18-f37072fd.pth"
main_pretrained = "multi-model.pth"
output_model = "late-fusion-model.pth"
config_file = "late-fusion-config.py"

resnet_state = torch.load(resnet_pretrained, map_location="cpu")
full_model = torch.load(main_pretrained, map_location="cpu")
full_state = full_model["state_dict"]
head_key = "keypoint_head.final_layer.weight"
full_state[head_key] = full_state[head_key][:, :32, :, :] * 1.5


for key in resnet_state.keys():
    if key.startswith("fc"):
        continue
    full_state["fusion_backbone." + key] = resnet_state[key]
full_state["fusion_head.2.weight"] = torch.randn(size=(3, 1024)) * 0.1
full_state["fusion_head.2.bias"] = torch.randn(size=(3,)) * 0.1

torch.save(full_model, output_model)

model = init_pose_model(config_file, output_model, device="cpu")

image_file = "/Users/alex/Downloads/person.jpg"
img_color = cv2.imread(image_file)
img_multi = np.zeros_like(img_color, shape=(img_color.shape[0], img_color.shape[1], 5))
img_multi[:, :, :3] = img_color * 0
img_multi[:, :, 3] = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_multi[:, :, 4] = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
inference_top_down_pose_model(model, img_multi)

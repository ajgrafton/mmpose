import numpy as np
import torch
from mmpose.apis import init_pose_model, inference_top_down_pose_model
import cv2


image_file = "/Users/alex/Downloads/person.jpg"
pretrained_model_file = "/Users/alex/Downloads/epoch_10.pth"
multi_model_file = "multi-model.pth"
multi_config_file = "multi-test-config.py"
ref_model_file = "/Users/alex/Downloads/td_torso_model.pth"
ref_model_config = "color-model-config.py"

model_params = torch.load(pretrained_model_file, map_location="cpu")

# The backbone needs to be replaced with "models.0" in the key
state_dict = model_params["state_dict"]
keys = list(state_dict.keys())
for k in keys:
    if not k.startswith("backbone"):
        continue
    new_key = "models.0" + k[8:]
    state_dict[new_key] = state_dict[k]

torch.save(model_params, "test_file.pth")

# Now do model 1 and 2
amount_added = 0
for k in keys:
    if not k.startswith("backbone"):
        continue
    for prefix in ["models.1", "models.2"]:
        new_key = prefix + k[8:]
        if k == "backbone.conv1.weight":
            state_dict[new_key] = torch.sum(state_dict[k], dim=1, keepdim=True)
            continue
        state_dict[new_key] = state_dict[k]
        amount_added += state_dict[new_key].numel()
print(amount_added)

# Make the keypoint head the right size
head_key = "keypoint_head.final_layer.weight"
state_dict[head_key] = torch.tile(state_dict[head_key], (1, 3, 1, 1)) / 1.5
# state_dict[head_key][:, 32:, :, :] = 0.0
# state_dict[head_key][:, :32, ...] = 0.0
# state_dict[head_key][:, 64:, ...] = 0.0

# Remove the original backbone
for k in keys:
    if not k.startswith("backbone"):
        continue
    state_dict.pop(k)
model_params["state_dict"] = state_dict

# Save and reload, along with a reference model
torch.save(model_params, multi_model_file)
# multi_model = init_pose_model(multi_config_file, multi_model_file, device="cpu")
multi_model = init_pose_model(multi_config_file, multi_model_file, device="cpu")
reference_model = init_pose_model(ref_model_config, ref_model_file, device="cpu")

# Create images to test
img_color = cv2.imread(image_file)
img_multi = np.zeros_like(img_color, shape=(img_color.shape[0], img_color.shape[1], 5))
img_multi[:, :, :3] = img_color * 0
img_multi[:, :, 3] = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_multi[:, :, 4] = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Test the models


results_multi = inference_top_down_pose_model(multi_model, img_multi)
results_ref = inference_top_down_pose_model(reference_model, img_color)

print(results_multi[0][0]["keypoints"])
print(results_ref[0][0]["keypoints"])

import torch
import numpy as np
from mmpose.apis import init_fusion_pose_model


resnet_18_rgb_path = "/Users/alex/Downloads/resnet18-f37072fd.pth"
resnet_18_1c_path = "resnet18_1c.pth"
resnet_18_5c_path = "resnet18_5c.pth"
model = torch.load(resnet_18_rgb_path, map_location="cpu")
model["conv1.weight"] = torch.sum(model["conv1.weight"], dim=1, keepdim=True)
torch.save(model, resnet_18_1c_path)
model["conv1.weight"] = torch.tile(model["conv1.weight"], (1, 5, 1, 1)) / 5.0
torch.save(model, resnet_18_5c_path)

# Adapt an early fusion model
early_fusion_ckpt = "/data/torso-models/fusion/checkpoints/epoch_2-v2.pth"
modified_ckpt = "/data/torso-models/fusion/checkpoints/epoch_2-v3.pth"
early_fusion_config = "/data/torso-models/fusion/early-fusion-stage2-config-5c.py"

full_model = torch.load(early_fusion_ckpt, map_location="cpu")
full_model["state_dict"]["fusion_backbone.conv1.weight"] = model["conv1.weight"]
torch.save(full_model, modified_ckpt)
loaded_model = init_fusion_pose_model(early_fusion_config, modified_ckpt, device="cpu")

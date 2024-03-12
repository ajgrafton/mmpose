import numpy as np
import torch
from mmpose.apis import init_pose_model, inference_top_down_pose_model
import cv2
from mmcv import Config

main_pretrained = "multi-model.pth"
output_model = "late-fusion-shared-model.pth"
config_file = "late-fusion-shared-config.py"
pretrained_hrnet = "/Users/alex/Downloads/td_torso_model.pth"

hrnet_state = torch.load(pretrained_hrnet, map_location="cpu")["state_dict"]
full_model = torch.load(main_pretrained, map_location="cpu")
full_state = full_model["state_dict"]
head_key = "keypoint_head.final_layer.weight"
full_state[head_key] = full_state[head_key][:, :32, :, :] * 1.5
config = Config.fromfile(config_file)

fuse_after_index = config["model"]["divide_after_stage"]
selector_size = config["model"]["selector_head_map_size"]
selector_model_indices = config["model"]["selector_model_indices"]


# Load a pre-trained HRNet but only the parts that we need!
# Use transition{fuse_after_index+}
# Use stage{fuse_after_index+1+}
hrnet_keys = hrnet_state.keys()
for key in hrnet_keys:
    if key.startswith("backbone.transition"):
        sub_key = key[19:]
        index = int(sub_key[: sub_key.find(".")])
        if index >= fuse_after_index:
            full_state["fusion_backbone." + key[9:]] = hrnet_state[key]
    if key.startswith("backbone.stage"):
        sub_key = key[14:]
        index = int(sub_key[: sub_key.find(".")])
        if index >= fuse_after_index + 1:
            full_state["fusion_backbone." + key[9:]] = hrnet_state[key]
full_state[
    f"fusion_backbone.transition{fuse_after_index}.{fuse_after_index}.0.0.weight"
] = torch.tile(
    full_state[
        f"fusion_backbone.transition{fuse_after_index}.{fuse_after_index}.0.0.weight"
    ],
    (1, len(selector_model_indices), 1, 1),
)
full_state[f"fusion_backbone.stage{fuse_after_index+1}.0.branches.0.0.conv1.weight"] = (
    torch.tile(
        full_state[
            f"fusion_backbone.stage{fuse_after_index+1}.0.branches.0.0.conv1.weight"
        ],
        (1, len(selector_model_indices), 1, 1),
    )
)
full_state[f"fusion_backbone.stage{fuse_after_index+1}.0.branches.1.0.conv1.weight"] = (
    torch.tile(
        full_state[
            f"fusion_backbone.stage{fuse_after_index+1}.0.branches.1.0.conv1.weight"
        ],
        (1, len(selector_model_indices), 1, 1),
    )
)

# MAKE THESE PROGRAMMATICALLY!
full_state[
    f"fusion_backbone.stage{fuse_after_index+1}.0.branches.0.0.downsample.0.weight"
] = (torch.randn(size=(32, 96, 1, 1)) * 0.1)
full_state[
    f"fusion_backbone.stage{fuse_after_index+1}.0.branches.0.0.downsample.1.weight"
] = (torch.randn(size=(32,)) * 0.1)
full_state[
    f"fusion_backbone.stage{fuse_after_index+1}.0.branches.0.0.downsample.1.bias"
] = (torch.randn(size=(32,)) * 0.1)
full_state[
    f"fusion_backbone.stage{fuse_after_index+1}.0.branches.0.0.downsample.1.running_mean"
] = (torch.randn(size=(32,)) * 0.1)
full_state[
    f"fusion_backbone.stage{fuse_after_index+1}.0.branches.0.0.downsample.1.running_var"
] = (torch.randn(size=(32,)) * 0.1)

full_state[
    f"fusion_backbone.stage{fuse_after_index+1}.0.branches.1.0.downsample.0.weight"
] = (torch.randn(size=(64, 192, 1, 1)) * 0.1)
full_state[
    f"fusion_backbone.stage{fuse_after_index+1}.0.branches.1.0.downsample.1.weight"
] = (torch.randn(size=(64,)) * 0.1)
full_state[
    f"fusion_backbone.stage{fuse_after_index+1}.0.branches.1.0.downsample.1.bias"
] = (torch.randn(size=(64,)) * 0.1)
full_state[
    f"fusion_backbone.stage{fuse_after_index+1}.0.branches.1.0.downsample.1.running_mean"
] = (torch.randn(size=(64,)) * 0.1)
full_state[
    f"fusion_backbone.stage{fuse_after_index+1}.0.branches.1.0.downsample.1.running_var"
] = (torch.randn(size=(64,)) * 0.1)


full_state["fusion_head.2.weight"] = (
    torch.randn(size=(3, selector_size[0] * selector_size[1] * 32 // 16)) * 0.1
)
full_state["fusion_head.2.bias"] = torch.randn(size=(3,)) * 0.1

# for key in hrnet_state.keys():
#     if key.startswith("fc"):
#         continue
#     full_state["fusion_backbone." + key] = resnet_state[key]
# full_state["fusion_head.2.weight"] = torch.randn(size=(3, 1024)) * 0.1
# full_state["fusion_head.2.bias"] = torch.randn(size=(3,)) * 0.1

torch.save(full_model, output_model)

model = init_pose_model(config_file, output_model, device="cpu")

image_file = "/Users/alex/Downloads/person.jpg"
img_color = cv2.imread(image_file)
img_multi = np.zeros_like(img_color, shape=(img_color.shape[0], img_color.shape[1], 5))
img_multi[:, :, :3] = img_color * 0
img_multi[:, :, 3] = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_multi[:, :, 4] = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
inference_top_down_pose_model(model, img_multi)

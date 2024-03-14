import numpy as np
import torch
from mmpose.apis import init_pose_model, inference_top_down_pose_model
import cv2
from mmcv import Config

main_pretrained = "multi-model.pth"
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

output_model = f"late-fusion-shared-stage{fuse_after_index}-model.pth"
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


if f"fusion_backbone.transition{fuse_after_index}.0.0.weight" in full_state:
    full_state[f"fusion_backbone.transition{fuse_after_index}.0.0.weight"] = torch.tile(
        full_state[f"fusion_backbone.transition{fuse_after_index}.0.0.weight"],
        (1, len(selector_model_indices), 1, 1),
    )

full_state[
    f"fusion_backbone.transition{fuse_after_index}.{fuse_after_index}.0.0.weight"
] = torch.tile(
    full_state[
        f"fusion_backbone.transition{fuse_after_index}.{fuse_after_index}.0.0.weight"
    ],
    (1, len(selector_model_indices), 1, 1),
)

for i in range(fuse_after_index):
    if fuse_after_index > 1:  # Otherwise the transition layer does this
        full_state[
            f"fusion_backbone.stage{fuse_after_index+1}.0.branches.{i}.0.conv1.weight"
        ] = torch.tile(
            full_state[
                f"fusion_backbone.stage{fuse_after_index+1}.0.branches.{i}.0.conv1.weight"
            ],
            (1, len(selector_model_indices), 1, 1),
        )

for j in range(fuse_after_index):
    if fuse_after_index == 1:
        break
    channels_after = config["model"]["selector"]["extra"][f"stage{fuse_after_index+1}"][
        "num_channels"
    ][j]
    channels_before = [
        model["extra"][f"stage{fuse_after_index+1}"]["num_channels"][j]
        for model in config["model"]["backbones"]
    ]
    full_state[
        f"fusion_backbone.stage{fuse_after_index+1}.0.branches.{j}.0.downsample.0.weight"
    ] = torch.zeros(size=(channels_after, sum(channels_before), 1, 1))
    start = 0
    for i in range(len(channels_before)):
        dim = min(channels_before[i], channels_after)
        full_state[
            f"fusion_backbone.stage{fuse_after_index + 1}.0.branches.{j}.0.downsample.0.weight"
        ][:dim, start : (start + dim), 0, 0] = torch.eye(dim) / len(channels_before)
        start += channels_before[i]

    full_state[
        f"fusion_backbone.stage{fuse_after_index + 1}.0.branches.{j}.0.downsample.1.weight"
    ] = torch.ones(size=(channels_after,))
    full_state[
        f"fusion_backbone.stage{fuse_after_index + 1}.0.branches.{j}.0.downsample.1.bias"
    ] = torch.zeros(size=(channels_after,))
    full_state[
        f"fusion_backbone.stage{fuse_after_index + 1}.0.branches.{j}.0.downsample.1.running_mean"
    ] = torch.zeros(size=(channels_after,))
    full_state[
        f"fusion_backbone.stage{fuse_after_index + 1}.0.branches.{j}.0.downsample.1.running_var"
    ] = torch.ones(size=(channels_after,))

full_state["fusion_head.2.weight"] = (
    torch.randn(size=(3, selector_size[0] * selector_size[1] * 32 // 16)) * 0.1
)
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

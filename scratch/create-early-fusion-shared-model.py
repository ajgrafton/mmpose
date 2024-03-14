import numpy as np
import torch
from mmpose.apis import init_pose_model, inference_top_down_pose_model
import cv2
from mmcv import Config

main_pretrained = "multi-model.pth"
output_model = "early-fusion-shared-model.pth"
config_file = "early-fusion-shared-config.py"
pretrained_hrnet = "/Users/alex/Downloads/td_torso_model.pth"

hrnet_state = torch.load(pretrained_hrnet, map_location="cpu")["state_dict"]
full_model = torch.load(main_pretrained, map_location="cpu")
full_state = full_model["state_dict"]

head_key = "keypoint_head.final_layer.weight"
full_state[head_key] = full_state[head_key][:, :32, :, :] * 1.5

config = Config.fromfile(config_file)

fusion_stage = config["model"]["fuse_after_stage"]
selector_size = config["model"]["selector_head_map_size"]
selector_model_indices = config["model"]["selector_model_indices"]

hrnet_keys = hrnet_state.keys()
state_keys = list(full_state.keys())

# ===== Remove the latter part of the HRNet (post-fusion)
# Remove models.{1+}.transition{fusion_stage+}...
# Remove models.{1+}.stage{fusion_stage+1+}...
for key in state_keys:
    split_key = key.split(".")
    if len(split_key) < 3:
        continue
    if split_key[0] == "models" and int(split_key[1]) >= 1:
        if (
            split_key[2].startswith("transition")
            and int(split_key[2][10:]) >= fusion_stage
        ):
            del full_state[key]
        if split_key[2].startswith("stage") and int(split_key[2][5:]) > fusion_stage:
            del full_state[key]

# ===== Add in the part of the pre-trained HRNet =====
for key in hrnet_keys:
    if key.startswith("backbone.transition"):
        sub_key = key[19:]
        index = int(sub_key[: sub_key.find(".")])
        if index >= fusion_stage:
            full_state["fusion_backbone." + key[9:]] = hrnet_state[key]
    if key.startswith("backbone.stage"):
        sub_key = key[14:]
        index = int(sub_key[: sub_key.find(".")])
        if index >= fusion_stage + 1:
            full_state["fusion_backbone." + key[9:]] = hrnet_state[key]

# ===== Modify the transition layer =====
if f"fusion_backbone.transition{fusion_stage}.0.0.weight" in full_state:
    full_state[f"fusion_backbone.transition{fusion_stage}.0.0.weight"] = torch.tile(
        full_state[f"fusion_backbone.transition{fusion_stage}.0.0.weight"],
        (1, len(selector_model_indices), 1, 1),
    )
full_state[f"fusion_backbone.transition{fusion_stage}.{fusion_stage}.0.0.weight"] = (
    torch.tile(
        full_state[
            f"fusion_backbone.transition{fusion_stage}.{fusion_stage}.0.0.weight"
        ],
        (1, len(selector_model_indices), 1, 1),
    )
)

# ===== Modify the stages in the fusion backbone as they now have more channels =====
n_branches = config["model"]["selector"]["extra"][f"stage{fusion_stage + 1}"][
    "num_branches"
]
for i in range(n_branches - 1):
    if fusion_stage == 1:
        break  # Transition layer does this
    if fusion_stage > 1:
        full_state[
            f"fusion_backbone.stage{fusion_stage+1}.0.branches.{i}.0.conv1.weight"
        ] = torch.tile(
            full_state[
                f"fusion_backbone.stage{fusion_stage+1}.0.branches.{i}.0.conv1.weight"
            ],
            (1, len(selector_model_indices), 1, 1),
        )

# ===== Create the downsample layers =====
for j in range(fusion_stage):
    if fusion_stage == 1:
        break
    channels_after = config["model"]["selector"]["extra"][f"stage{fusion_stage+1}"][
        "num_channels"
    ][j]
    channels_before = [
        model["extra"][f"stage{fusion_stage+1}"]["num_channels"][j]
        for model in config["model"]["backbones"]
    ]
    full_state[
        f"fusion_backbone.stage{fusion_stage+1}.0.branches.{j}.0.downsample.0.weight"
    ] = torch.zeros(size=(channels_after, sum(channels_before), 1, 1))
    start = 0
    for i in range(len(channels_before)):
        dim = min(channels_before[i], channels_after)
        full_state[
            f"fusion_backbone.stage{fusion_stage + 1}.0.branches.{j}.0.downsample.0.weight"
        ][:dim, start : (start + dim), 0, 0] = torch.eye(dim) / len(channels_before)
        start += channels_before[i]

    full_state[
        f"fusion_backbone.stage{fusion_stage + 1}.0.branches.{j}.0.downsample.1.weight"
    ] = torch.ones(size=(channels_after,))
    full_state[
        f"fusion_backbone.stage{fusion_stage + 1}.0.branches.{j}.0.downsample.1.bias"
    ] = torch.zeros(size=(channels_after,))
    full_state[
        f"fusion_backbone.stage{fusion_stage + 1}.0.branches.{j}.0.downsample.1.running_mean"
    ] = torch.zeros(size=(channels_after,))
    full_state[
        f"fusion_backbone.stage{fusion_stage + 1}.0.branches.{j}.0.downsample.1.running_var"
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

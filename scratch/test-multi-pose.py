import torch
from mmpose.apis import init_pose_model


img_1 = torch.zeros((1, 3, 960, 512), dtype=torch.float)
img_2 = torch.zeros((1, 1, 960, 512), dtype=torch.float)
img_3 = torch.zeros((1, 1, 960, 512), dtype=torch.float)
images = [img_1, img_2, img_3]
meta_dict = {"center": [480, 256], "scale": [8, 10], "image_file": ""}
target = torch.zeros((1, 4, 192, 160), dtype=torch.float)
target_weight = torch.zeros((1, 4, 1), dtype=torch.float)
metas = [meta_dict, meta_dict, meta_dict]
config_file = "scratch/multi-test-config.py"
model = init_pose_model(config_file, device="cpu")
result = model.forward(
    images,
    img_metas=metas,
    return_loss=True,
    target=target,
    target_weight=target_weight,
)
print(result)

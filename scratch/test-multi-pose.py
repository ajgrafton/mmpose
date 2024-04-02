from mmpose.apis import init_pose_model
from mmpose.core import optimizers
from mmcv.runner.epoch_based_runner import save_checkpoint
from mmcv import Config
import torch

device = torch.device("cpu")
device = "cpu"

img_1 = torch.zeros((1, 3, 480, 256), dtype=torch.float, device=device)
img_2 = torch.zeros((1, 1, 480, 256), dtype=torch.float, device=device)
img_3 = torch.zeros((1, 1, 480, 256), dtype=torch.float, device=device)
imgs = torch.cat([img_1, img_2, img_3], dim=1)
meta_dict = {"center": [240, 128], "scale": [8, 10], "image_file": ""}
target = torch.zeros((1, 4, 96, 80), dtype=torch.float, device=device)
target_weight = torch.zeros((1, 4, 1), dtype=torch.float, device=device)
metas = [meta_dict]
config_file = "multi-test-config.py"
model = init_pose_model(config_file, device="cpu")
save_checkpoint(model, "test.pth")
loaded = torch.load("test.pth")
state_dict = loaded["state_dict"]
for key in state_dict.keys():
    print(key)
exit()
config = Config.fromfile(config_file)
optimizer: torch.optim.Optimizer = optimizers.build_optimizers(
    model, config["optimizer"]
)
batch = {
    "img": torch.cat([imgs, imgs, imgs], dim=0),
    "img_metas": [metas, metas, metas],
    "return_loss": True,
    "target": torch.cat([target] * 3, dim=0),
    "target_weight": torch.cat([target_weight] * 3, dim=0),
}
# result = model.forward(
#     imgs,
#     img_metas=metas,
#     return_loss=True,
#     target=target,
#     target_weight=target_weight,
# )

print(model.train_step(batch, optimizer))

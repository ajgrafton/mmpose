import warnings

import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow
from mmpose.models.backbones import ResNet
from mmpose.core import imshow_bboxes, imshow_keypoints
from .. import builder
from ..builder import POSENETS
from .base import BasePose
from .top_down import TopDown
import torch
from typing import List


try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn(
        "auto_fp16 from mmpose will be deprecated from v0.15.0"
        "Please install mmcv>=1.1.4"
    )
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class TopDownLateFusion(BasePose):
    """
    Top down multiple image detector with late fusion to select which image source to use
    """

    def __init__(
        self,
        color_index,
        backbones,
        keypoint_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super().__init__()
        self.fp16_enabled = False
        self.models = torch.nn.ModuleList()
        self.model_slices = []
        self.fusion_selector_backbone = ResNet(18, in_channels=3)
        self.fusion_selector_head = self.make_selector_head()
        current_channel = 0
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.color_index = color_index

        self.num_models = len(backbones)
        for i in range(self.num_models):
            backbone_i = backbones[i]
            model_i = builder.build_backbone(backbone_i)
            num_channels_i = backbone_i["in_channels"]
            prefix = f"models.{i}."
            model_i.init_weights(pretrained, override_prefix=prefix)

            self.model_slices.append(
                slice(current_channel, current_channel + num_channels_i)
            )
            current_channel += num_channels_i
            self.models.append(model_i)
        keypoint_head["train_cfg"] = train_cfg
        keypoint_head["test_cfg"] = test_cfg
        self.keypoint_head = builder.build_head(keypoint_head)

    @staticmethod
    def make_selector_head():
        pool_layer = torch.nn.AvgPool2d(kernel_size=4)
        flatten_layer = torch.nn.Flatten()
        linear_layer = torch.nn.Linear(1024, 3)
        softmax_layer = torch.nn.Softmax(dim=1)
        return torch.nn.Sequential(
            pool_layer, flatten_layer, linear_layer, softmax_layer
        )

    @auto_fp16(apply_to=("img",))
    def forward(
        self,
        img,
        target=None,
        target_weight=None,
        img_metas=None,
        return_loss=True,
        return_heatmap=False,
        **kwargs,
    ):
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas, **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs
        )

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        # Get the outputs for the backbones
        sub_images = [img[:, model_slice, ...] for model_slice in self.model_slices]
        output = [self.models[i](sub_images[i]) for i in range(self.num_models)]

        # Stack the outputs
        output = [
            torch.cat([op[i] for op in output], dim=1) for i in range(len(output[0]))
        ]

        # Run through the head
        output = self.keypoint_head(output)

        # Get the losses
        losses = dict()
        keypoint_losses = self.keypoint_head.get_loss(output, target, target_weight)
        losses.update(keypoint_losses)
        keypoint_accuracy = self.keypoint_head.get_accuracy(
            output, target, target_weight
        )
        losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        assert len(img) == len(img_metas)
        sub_images = [img[:, model_slice, ...] for model_slice in self.model_slices]
        batch_size, _, img_height, img_width = sub_images[0].shape
        # assert len(sub_images) == self.num_models
        if batch_size > 1:
            assert "bbox_id" in img_metas[0]

        result = {}
        features = torch.stack(
            [self.models[i](sub_images[i])[0] for i in range(self.num_models)], dim=0
        )

        # Run the fusion backbone
        color_img = sub_images[self.color_index]
        fusion_backbone_result = self.fusion_selector_backbone(color_img)
        fusion_result = self.fusion_selector_head(fusion_backbone_result).reshape(
            [3, 1, 1, 1, 1]
        )
        features = torch.sum(features * fusion_result, dim=0, keepdim=False)

        # features = [
        #     torch.cat([op[i] for op in features], dim=1)
        #     for i in range(len(features[0]))
        # ]
        output_heatmap = self.keypoint_head.inference_model(features, flip_pairs=None)

        if self.test_cfg.get("flip_test", True):
            img_flipped = img.flip(4)
            sub_images_flipped = [
                img_flipped[:, model_slice, ...] for model_slice in self.model_slices
            ]
            features_flipped = torch.stack(
                [
                    self.models[i](sub_images_flipped[i])[0]
                    for i in range(self.num_models)
                ],
                dim=4,
            )
            color_img_flipped = sub_images_flipped[self.color_index]
            fusion_backbone_result_flipped = self.fusion_selector_backbone(
                color_img_flipped
            )
            fusion_result_flipped = self.fusion_selector_head(
                fusion_backbone_result_flipped
            ).reshape([3, 1, 1, 1, 1])
            features_flipped = torch.sum(
                features_flipped * fusion_result_flipped, dim=0, keepdim=False
            )

            # Stack the outputs
            # features_flipped = [
            #     torch.cat([op[i] for op in features_flipped], dim=0)
            #     for i in range(len(features_flipped[0]))
            # ]
            output_flipped_heatmap = self.keypoint_head.inference_model(
                features_flipped, img_metas[0]["flip_pairs"]
            )
            output_heatmap = output_heatmap + output_flipped_heatmap
            if self.test_cfg.get("regression_flip_shift", False):
                output_heatmap[..., 0] -= 1.0 / img_width
            output_heatmap = output_heatmap / 2

        keypoint_result = self.keypoint_head.decode(
            img_metas, output_heatmap, img_size=[img_width, img_height]
        )
        result.update(keypoint_result)

        if not return_heatmap:
            output_heatmap = None
        result["output_heatmap"] = output_heatmap
        return result

    def show_result(self, **kwargs):
        pass

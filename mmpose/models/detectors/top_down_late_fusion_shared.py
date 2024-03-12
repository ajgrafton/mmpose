import copy
import warnings

import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow
from mmpose.models.backbones import ResNet
from mmpose.core import imshow_bboxes, imshow_keypoints
import torchvision
from .. import builder
from ..backbones import BreakableBackbone
from ..builder import POSENETS
from .base import BasePose
from .top_down import TopDown
import torch
from typing import List, Dict, Union
from torch import Tensor


try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn(
        "auto_fp16 from mmpose will be deprecated from v0.15.0"
        "Please install mmcv>=1.1.4"
    )
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class TopDownLateFusionShared(BasePose):
    """
    Top down multiple image detector with late fusion to select which image source to use
    """

    def __init__(
        self,
        selector_model_indices,
        selector,
        backbones,
        keypoint_head,
        selector_head_map_size,
        divide_after_stage: int,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super().__init__()
        self.fp16_enabled = False
        self.models = torch.nn.ModuleList()
        self.model_slices = []
        self.num_models = len(backbones)
        self.divide_after_stage = divide_after_stage
        self.selector_head_map_size = [x for x in selector_head_map_size]
        self.pretrained = pretrained

        self.fusion_backbone = None
        self.fusion_head = None
        self.output_resizer = None
        self.build_selector(selector, backbones)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.selector_model_indices = selector_model_indices

        self.create_backbones(backbones)

        keypoint_head["train_cfg"] = train_cfg
        keypoint_head["test_cfg"] = test_cfg
        self.keypoint_head = builder.build_head(keypoint_head)

    @staticmethod
    def make_selector_head(linear_layer_size: int, num_fusion_weights: int):
        pool_layer = torch.nn.AvgPool2d(kernel_size=4)
        flatten_layer = torch.nn.Flatten()
        linear_layer = torch.nn.Linear(linear_layer_size // 16, num_fusion_weights)
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

    def apply_late_shared_fusion(
        self,
        part_1_features: List[Union[Tensor, List[Tensor]]],
        features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Runs the image through the fusion selector backbone and head, then uses
        the results to fuse the features
        Args:
            part_1_features: Input features that are the output from each part 1 model
            features: List of outputs from each backbone

        Returns: Features after fusion
        """

        # Prepare the features for the fusion backbone
        fusion_inputs = self.fusion_backbone.prep_inputs(part_1_features)
        # Run the fusion model (part 2 only)
        backbone_output = self.fusion_backbone(fusion_inputs, part=2)
        backbone_output = self.fusion_backbone.prep_output(backbone_output)
        # Resize the output so it's suitable for a Dense layer
        backbone_output = self.output_resizer(backbone_output)
        # Run the fusion head
        fusion_weights = self.fusion_head(backbone_output)
        fusion_weights = fusion_weights.reshape([self.num_models, 1, 1, 1, 1])
        # Stack the features and do the weighted sum
        features = torch.stack(features, dim=0)
        fused_features = torch.sum(features * fusion_weights, dim=0, keepdim=False)
        return fused_features

    def divide_into_sub_images(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        Divides the input image into sub-images for each backbone
        Args:
            img: Input image

        Returns: List of sub-images for each backbone (List[Tensor])

        """
        sub_images = [img[:, model_slice, ...] for model_slice in self.model_slices]
        return sub_images

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        # Get the outputs for the backbones
        sub_images = self.divide_into_sub_images(img)

        part_1_features = [
            self.models[i](sub_images[i], part=1) for i in range(self.num_models)
        ]

        part_2_features = [
            self.models[i].prep_output(self.models[i](part_1_features[i], part=2))
            for i in range(self.num_models)
        ]

        fused_features = self.apply_late_shared_fusion(part_1_features, part_2_features)

        output = self.keypoint_head(fused_features)

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
        sub_images = self.divide_into_sub_images(img)
        batch_size, _, img_height, img_width = sub_images[0].shape
        if batch_size > 1:
            assert "bbox_id" in img_metas[0]
        result = {}

        # Run part 1 of each backbone
        part_1_features = [
            self.models[i](sub_images[i], part=1) for i in range(self.num_models)
        ]
        # Run part 2 of all the other models
        part_2_features = [
            self.models[i].prep_output(self.models[i](part_1_features[i], part=2))
            for i in range(self.num_models)
        ]
        # Apply the fusion
        fused_features = self.apply_late_shared_fusion(part_1_features, part_2_features)
        # Run through the keypoint head
        output_heatmap = self.keypoint_head.inference_model(
            fused_features, flip_pairs=None
        )

        if self.test_cfg.get("flip_test", True):
            # Flip the image
            img_flipped = img.flip(3)
            sub_images_flipped = self.divide_into_sub_images(img_flipped)
            part_1_flipped = [
                self.models[i](sub_images_flipped[i], part=1)
                for i in range(self.num_models)
            ]
            part_2_flipped = [
                self.models[i].prep_output(self.models[i](part_1_flipped[i], part=2))
                for i in range(self.num_models)
            ]
            fused_flipped = self.apply_late_shared_fusion(
                part_1_flipped, part_2_flipped
            )
            output_heatmap_flipped = self.keypoint_head.inference_model(
                fused_flipped, flip_pairs=img_metas[0]["flip_pairs"]
            )

            # Combine flipped and un-flipped
            output_heatmap = output_heatmap + output_heatmap_flipped
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

    def build_selector(self, selector, backbones):
        # Set the configuration for the selector backbone
        selector["divide_after_stage"] = self.divide_after_stage
        selector["run_part_1"] = False
        selector["run_part_2"] = True
        selector["inputs_to_combine"] = backbones
        self.fusion_backbone = builder.build_backbone(selector)
        if not isinstance(self.fusion_backbone, BreakableBackbone):
            raise ValueError("Fusion selector backbone should be breakable")

        # Create the output resizer, before the dense layer
        linear_layer_size = (
            self.fusion_backbone.num_output_channels()
            * self.selector_head_map_size[0]
            * self.selector_head_map_size[1]
        )
        self.output_resizer = torchvision.transforms.Resize(
            (self.selector_head_map_size[1], self.selector_head_map_size[0]),
            antialias=False,
        )

        # Then build the fusion selector head
        self.fusion_head = self.make_selector_head(linear_layer_size, self.num_models)

    def create_backbones(self, backbones: List[Dict]):
        current_channel = 0
        for i in range(self.num_models):
            backbone_i = copy.deepcopy(backbones[i])
            backbone_i["divide_after_stage"] = self.divide_after_stage
            backbone_i["run_part_1"] = True
            backbone_i["run_part_2"] = True
            model_i = builder.build_backbone(backbone_i)
            num_channels_i = backbone_i["in_channels"]
            prefix = f"models.{i}."
            model_i.init_weights(self.pretrained, override_prefix=prefix)

            self.model_slices.append(
                slice(current_channel, current_channel + num_channels_i)
            )
            current_channel += num_channels_i
            self.models.append(model_i)

    def show_result(self, **kwargs):
        pass

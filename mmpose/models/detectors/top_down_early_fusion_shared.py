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
class TopDownEarlyFusionShared(BasePose):
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
        fuse_after_stage: int,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super().__init__()
        self.fp16_enabled = False
        self.models = torch.nn.ModuleList()
        self.model_slices = []
        self.num_models = len(backbones)
        self.fuse_after_stage = fuse_after_stage
        self.pretrained = pretrained
        self.selector_head_map_size = [x for x in selector_head_map_size]
        self.selector_model_indices = selector_model_indices
        self.fusion_only = False
        self.fusion_backbone = None
        self.fusion_head = None
        self.output_resizer = None

        self.build_selector(selector, backbones)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.create_backbones(backbones)

        keypoint_head["train_cfg"] = train_cfg
        keypoint_head["test_cfg"] = test_cfg
        self.keypoint_head = builder.build_head(keypoint_head)

    def set_fusion_only(self, fusion_only: bool = True) -> None:
        self.fusion_only = fusion_only

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
        if self.fusion_only:
            return self.forward_fusion(img, img_metas)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs
        )

    def forward_fusion(self, img, img_metas):
        assert len(img) == len(img_metas)
        sub_images = self.divide_into_sub_images(img)
        batch_size, _, img_height, img_width = sub_images[0].shape
        if batch_size > 1:
            assert "bbox_id" in img_metas[0]
        result = {}

        part_1_features = [
            self.models[i](sub_images[i], part=1) for i in range(self.num_models)
        ]
        input_features = [part_1_features[i] for i in self.selector_model_indices]
        fusion_inputs = self.fusion_backbone.prep_inputs(input_features)
        backbone_output = self.fusion_backbone(fusion_inputs, part=2)
        backbone_output = self.fusion_backbone.prep_output(backbone_output)
        backbone_output = self.output_resizer(backbone_output)
        fusion_weights = self.fusion_head(backbone_output)
        result["weights"] = fusion_weights
        return result

    def apply_early_fusion(
        self, part_1_features: Union[List[Tensor], List[List[Tensor]]]
    ) -> Union[Tensor, List[Tensor]]:
        list_of_lists = True
        if isinstance(part_1_features[0], Tensor):
            list_of_lists = False

        # Put the results of part 1 through the fusion backbone
        input_features = [part_1_features[i] for i in self.selector_model_indices]
        fusion_inputs = self.fusion_backbone.prep_inputs(input_features)
        backbone_output = self.fusion_backbone(fusion_inputs, part=2)
        backbone_output = self.fusion_backbone.prep_output(backbone_output)
        backbone_output = self.output_resizer(backbone_output)
        fusion_weights = self.fusion_head(backbone_output)
        fusion_weights = torch.transpose(fusion_weights, 0, 1).reshape(
            [self.num_models, -1, 1, 1, 1]
        )

        if list_of_lists:
            results = []
            for i in range(len(part_1_features[0])):
                stacked_features = torch.stack([p[i] for p in part_1_features], dim=0)
                fused_features = torch.sum(
                    stacked_features * fusion_weights, dim=0, keepdim=False
                )
                results.append(fused_features)
            return results

        stacked_features = torch.stack(part_1_features, dim=0)
        fused_features = torch.sum(
            stacked_features * fusion_weights, dim=0, keepdim=False
        )
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

        # Do the signal fusion
        fused_part_1 = self.apply_early_fusion(part_1_features)

        # Do part 2 of the first model only, and head
        part_2_result = self.models[0](fused_part_1, part=2)

        output = self.keypoint_head(part_2_result)

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

        # Do the signal fusion
        fused_part_1 = self.apply_early_fusion(part_1_features)

        # Do part 2 of the first model only, and head
        part_2_result = self.models[0](fused_part_1, part=2)
        output_heatmap = self.keypoint_head.inference_model(
            part_2_result, flip_pairs=None
        )

        if self.test_cfg.get("flip_test", True):
            # Flip the image
            img_flipped = img.flip(3)
            sub_images_flipped = self.divide_into_sub_images(img_flipped)
            part_1_flipped = [
                self.models[i](sub_images_flipped[i], part=1)
                for i in range(self.num_models)
            ]
            fused_flipped_1 = self.apply_early_fusion(part_1_flipped)
            part_2_flipped = self.models[0](fused_flipped_1, part=2)
            output_heatmap_flipped = self.keypoint_head.inference_model(
                part_2_flipped, flip_pairs=img_metas[0]["flip_pairs"]
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

    def make_selector_head(self, linear_layer_size: int):
        pool_layer = torch.nn.AvgPool2d(kernel_size=4)
        flatten_layer = torch.nn.Flatten()
        linear_layer = torch.nn.Linear(linear_layer_size // 16, self.num_models)
        softmax_layer = torch.nn.Softmax(dim=1)
        return torch.nn.Sequential(
            pool_layer, flatten_layer, linear_layer, softmax_layer
        )

    def build_selector(self, selector, backbones):
        selector["divide_after_stage"] = self.fuse_after_stage
        selector["run_part_1"] = False
        selector["run_part_2"] = True
        selector["inputs_to_combine"] = [
            backbones[i] for i in self.selector_model_indices
        ]
        self.fusion_backbone = builder.build_backbone(selector)
        if not isinstance(self.fusion_backbone, BreakableBackbone):
            raise ValueError("Fusion selector backbone should be breakable")

        linear_layer_size = (
            self.fusion_backbone.num_output_channels()
            * self.selector_head_map_size[0]
            * self.selector_head_map_size[1]
        )
        self.output_resizer = torchvision.transforms.Resize(
            (self.selector_head_map_size[1], self.selector_head_map_size[0]),
            antialias=False,
        )

        self.fusion_head = self.make_selector_head(linear_layer_size)

    def create_backbones(self, backbones: List[Dict]):
        current_channel = 0
        for i in range(self.num_models):
            backbone_i = copy.deepcopy(backbones[i])
            backbone_i["divide_after_stage"] = self.fuse_after_stage
            backbone_i["run_part_1"] = True
            if i == 0:
                backbone_i["run_part_2"] = True
            else:
                backbone_i["run_part_2"] = False
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

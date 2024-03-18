import warnings

import mmcv
import numpy as np
import torchvision
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
        selector_indices,
        selector,
        backbones,
        keypoint_head,
        selector_head_map_size,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super().__init__()
        self.fp16_enabled = False
        self.models = torch.nn.ModuleList()
        self.model_slices = []
        self.selector_indices = selector_indices
        selector["in_channels"] = len(self.selector_indices)
        self.selector_head_map_size = [x for x in selector_head_map_size]
        self.output_resizer = None
        self.fusion_backbone = builder.build_backbone(selector)
        self.fusion_head = self.make_selector_head()
        current_channel = 0
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

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

    def make_selector_head(self):
        self.output_resizer = torchvision.transforms.Resize(
            (self.selector_head_map_size[1], self.selector_head_map_size[0]),
            antialias=False,
        )
        linear_layer_size = (
            self.selector_head_map_size[0] * self.selector_head_map_size[1] * 8
        )
        pool_layer = torch.nn.AvgPool2d(kernel_size=8)
        flatten_layer = torch.nn.Flatten()
        linear_layer = torch.nn.Linear(linear_layer_size, 3)
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

    def apply_late_fusion(
        self, img: torch.Tensor, features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Runs the image through the fusion selector backbone and head, then uses
        the results to fuse the features
        Args:
            img: Input image
            features: List of outputs from each backbone

        Returns: Features after fusion
        """

        # Run the image through the head and backbone, using only the channels
        # relevant for the fusion selection, as specified by self.selector_indices
        # in the config.
        backbone_result = self.fusion_backbone(img[:, self.selector_indices, ...])
        fusion_result = self.fusion_head(backbone_result)

        # Reshape the fusion result so that it can be applied to the features tensor:
        # [num models x num images x num feature maps x height x width
        fusion_result = fusion_result.reshape([self.num_models, -1, 1, 1, 1])
        print(f"Fusion Weights = {fusion_result.reshape([-1])}")
        # Stack up the features and use the fusion results as a weighed sum
        stacked_features = torch.stack(features, dim=0)
        fused_features = torch.sum(
            stacked_features * fusion_result, dim=0, keepdim=False
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
        # sub_images = [img[:, model_slice, ...] for model_slice in self.model_slices]
        features = [self.models[i](sub_images[i])[0] for i in range(self.num_models)]

        # Stack the outputs
        # output = [
        #     torch.cat([op[i] for op in output], dim=1) for i in range(len(output[0]))
        # ]
        fused_features = self.apply_late_fusion(img, features)

        # Run through the head
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

        # Run backbones
        features = [self.models[i](sub_images[i])[0] for i in range(self.num_models)]

        # Run fusion
        fused_features = self.apply_late_fusion(img, features)

        # Run keypoint head
        output_heatmap = self.keypoint_head.inference_model(
            fused_features, flip_pairs=None
        )

        if self.test_cfg.get("flip_test", True):
            # Flip the image
            img_flipped = img.flip(3)
            sub_images_flipped = self.divide_into_sub_images(img_flipped)

            # Run backbones
            features_flipped = [
                self.models[i](sub_images_flipped[i])[0] for i in range(self.num_models)
            ]

            # Run fusion
            fused_features_flipped = self.apply_late_fusion(
                img_flipped, features_flipped
            )

            # Run keypoint head
            output_flipped_heatmap = self.keypoint_head.inference_model(
                fused_features_flipped, img_metas[0]["flip_pairs"]
            )

            # Combine flipped and un-flipped
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

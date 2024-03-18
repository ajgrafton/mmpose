import warnings

import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

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
class MultiTopDown(BasePose):
    """
    Multiple image pose detector lets goooo
    """

    def __init__(
        self,
        backbones,
        keypoint_head,
        necks=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        loss_pose=None,
    ):
        super().__init__()
        self.fp16_enabled = False
        self.models = torch.nn.ModuleList()
        self.model_slices = []
        current_channel = 0
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_models = len(backbones)
        for i in range(self.num_models):
            backbone_i = backbones[i]
            # neck_i = None
            # if necks is not None:
            # neck_i = builder.build_neck(necks[i])

            model_i = builder.build_backbone(backbone_i)
            num_channels_i = backbone_i["in_channels"]
            prefix = f"models.{i}."
            # pretrained_i = None
            # if pretrained is not None:
            # pretrained_i = pretrained[i]
            model_i.init_weights(pretrained, override_prefix=prefix)

            # model_i = TopDown(
            #     backbone_i, neck_i, None, train_cfg, test_cfg, pretrained_i, loss_pose
            # )
            # model_i.init_weights(None)
            self.model_slices.append(
                slice(current_channel, current_channel + num_channels_i)
            )
            # model_i = model_i.to('cuda')
            self.models.append(model_i)
            current_channel += num_channels_i

        keypoint_head["train_cfg"] = train_cfg
        keypoint_head["test_cfg"] = test_cfg
        self.keypoint_head = builder.build_head(keypoint_head)

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
        features = [self.models[i](sub_images[i]) for i in range(self.num_models)]

        # Stack the outputs
        features = [
            torch.cat([op[i] for op in features], dim=1)
            for i in range(len(features[0]))
        ]
        output_heatmap = self.keypoint_head.inference_model(features, flip_pairs=None)

        if self.test_cfg.get("flip_test", True):
            img_flipped = img.flip(3)
            sub_images_flipped = [
                img_flipped[:, model_slice, ...] for model_slice in self.model_slices
            ]
            features_flipped = [
                self.models[i](sub_images_flipped[i]) for i in range(self.num_models)
            ]

            # Stack the outputs
            features_flipped = [
                torch.cat([op[i] for op in features_flipped], dim=1)
                for i in range(len(features_flipped[0]))
            ]
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

    def forward_dummy(self, img):
        dummy_results = [
            self.models[i].forward_dummy(img[i]) for i in range(self.num_models)
        ]
        dummy_results = torch.cat(dummy_results, dim=0)
        dummy_results = self.keypoint_head(dummy_results)
        return dummy_results

    @deprecated_api_warning({"pose_limb_color": "pose_link_color"}, cls_name="TopDown")
    def show_result(
        self,
        img,
        result,
        skeleton=None,
        kpt_score_thr=0.3,
        bbox_color="green",
        pose_kpt_color=None,
        pose_link_color=None,
        text_color="white",
        radius=4,
        thickness=1,
        font_scale=0.5,
        bbox_thickness=1,
        win_name="",
        show=False,
        show_keypoint_weight=False,
        wait_time=0,
        out_file=None,
    ):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if "bbox" in res:
                bbox_result.append(res["bbox"])
                bbox_labels.append(res.get("label", None))
            pose_result.append(res["keypoints"])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False,
            )

        if pose_result:
            imshow_keypoints(
                img,
                pose_result,
                skeleton,
                kpt_score_thr,
                pose_kpt_color,
                pose_link_color,
                radius,
                thickness,
            )

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img

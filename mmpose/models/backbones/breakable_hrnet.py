import copy

import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Union, Dict
from mmcv.cnn import build_conv_layer, build_norm_layer, constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .resnet import BasicBlock, Bottleneck, get_expansion
from .utils import load_checkpoint
import numpy as np

from .hrnet import HRModule, HRNet
from .breakable_backbone import BreakableBackbone


@BACKBONES.register_module()
class BreakableHRNet(nn.Module, BreakableBackbone):
    def __init__(
        self,
        extra,
        divide_after_stage: int,
        inputs_to_combine: List[Dict] = None,
        run_part_1: bool = True,
        run_part_2: bool = True,
        frozen_stages: int = -1,
        in_channels=3,
        conv_cfg=None,
        norm_cfg=None,
        norm_eval=False,
        with_cp=False,
        zero_init_residual=False,
        stage_1_noise=None,
        stage_2_noise=None,
        stage_3_noise=None,
        dropout_1=None,
        dropout_2=None,
        dropout_3=None,
    ):
        if norm_cfg is None:
            norm_cfg = dict(type="BN")
        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.norm_eval = norm_eval
        self.conv_cfg = conv_cfg
        self.extra = extra
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.frozen_stages = frozen_stages
        super().__init__()

        if (
            stage_1_noise is not None
            or stage_2_noise is not None
            or stage_3_noise is not None
        ):
            raise NotImplementedError("Noise is not yet implemented for BreakableHRNet")
        if dropout_1 is not None or dropout_2 is not None or dropout_3 is not None:
            raise NotImplementedError(
                "Dropout is not yet implemented for BreakableHRNet"
            )

        self.run_part_1 = run_part_1
        self.run_part_2 = run_part_2
        self.divide_stage = divide_after_stage
        self.build_stage_1 = False
        self.build_stage_2 = False
        self.build_stage_3 = False
        self.build_stage_4 = False
        self.decide_which_channels_to_use()
        self.inputs_to_combine = inputs_to_combine
        self.extra["stage1"]["num_channels_combined"] = None
        self.extra["stage2"]["num_channels_combined"] = None
        self.extra["stage3"]["num_channels_combined"] = None
        if self.inputs_to_combine is not None:
            self._modify_config()

        self.stage1_cfg = self.extra["stage1"]
        self.stage2_cfg = self.extra["stage2"]
        self.stage3_cfg = self.extra["stage3"]
        self.stage4_cfg = self.extra["stage4"]
        self.upsample_cfg = self.extra.get(
            "upsample", {"mode": "nearest", "align_corners": None}
        )

        if self.build_stage_1:
            self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
            self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

            self.add_module(self.norm1_name, norm1)
            self.conv2 = build_conv_layer(
                self.conv_cfg, 64, 64, kernel_size=3, stride=2, padding=1, bias=False
            )

            self.add_module(self.norm2_name, norm2)
            self.relu = nn.ReLU(inplace=True)

            self.upsample_cfg = self.extra.get(
                "upsample", {"mode": "nearest", "align_corners": None}
            )

            # stage 1
            num_channels = self.stage1_cfg["num_channels"][0]
            block_type = self.stage1_cfg["block"]
            num_blocks = self.stage1_cfg["num_blocks"][0]

            block = HRNet.blocks_dict[block_type]
            stage1_out_channels = num_channels * get_expansion(block)
            self.layer1 = self._make_layer(block, 64, stage1_out_channels, num_blocks)

        else:
            # need to produce the value stage1_out_channels somehow
            num_channels = self.stage1_cfg["num_channels"][0]
            block_type = self.stage1_cfg["block"]
            block = HRNet.blocks_dict[block_type]
            stage1_out_channels = num_channels * get_expansion(block)

        if self.build_stage_2:
            # stage 2
            num_channels = self.stage2_cfg["num_channels"]
            block_type = self.stage2_cfg["block"]

            block = HRNet.blocks_dict[block_type]
            num_channels = [channel * get_expansion(block) for channel in num_channels]
            self.transition1, transition1_channels = self._make_transition_layer(
                [stage1_out_channels],
                num_channels,
                self.stage1_cfg["num_channels_combined"],
            )
            self.stage2, pre_stage_channels = self._make_stage(
                self.stage2_cfg, transition1_channels
            )
        else:
            # Need to calculate pre-stage-channels
            num_channels = self.stage2_cfg["num_channels"]
            block_type = self.stage2_cfg["block"]
            block = HRNet.blocks_dict[block_type]
            pre_stage_channels = [
                channel * get_expansion(block) for channel in num_channels
            ]

        if self.build_stage_3:
            # stage 3
            num_channels = self.stage3_cfg["num_channels"]
            block_type = self.stage3_cfg["block"]

            block = HRNet.blocks_dict[block_type]
            num_channels = [channel * get_expansion(block) for channel in num_channels]
            self.transition2, transition2_channels = self._make_transition_layer(
                pre_stage_channels,
                num_channels,
                self.stage2_cfg["num_channels_combined"],
            )
            self.stage3, pre_stage_channels = self._make_stage(
                self.stage3_cfg, transition2_channels
            )
        else:
            # need to calculate pre_stage_channels
            num_channels = self.stage3_cfg["num_channels"]
            block_type = self.stage3_cfg["block"]
            block = HRNet.blocks_dict[block_type]
            num_channels = [channel * get_expansion(block) for channel in num_channels]
            pre_stage_channels = num_channels

        if self.build_stage_4:
            # stage 4
            num_channels = self.stage4_cfg["num_channels"]
            block_type = self.stage4_cfg["block"]

            block = HRNet.blocks_dict[block_type]
            num_channels = [channel * get_expansion(block) for channel in num_channels]
            self.transition3, transition3_channels = self._make_transition_layer(
                pre_stage_channels,
                num_channels,
                self.stage3_cfg["num_channels_combined"],
            )

            self.stage4, pre_stage_channels = self._make_stage(
                self.stage4_cfg,
                transition3_channels,
                multiscale_output=self.stage4_cfg.get("multiscale_output", False),
            )

    def _modify_config(self):
        if self.divide_stage == 1:
            self.extra["stage1"]["num_channels_combined"] = (
                sum(
                    [
                        backbone["extra"]["stage1"]["num_channels"][0]
                        * get_expansion(
                            HRNet.blocks_dict[backbone["extra"]["stage1"]["block"]]
                        )
                        for backbone in self.inputs_to_combine
                    ]
                ),
            )
            return
        elif self.divide_stage == 2:
            self.extra["stage2"]["num_channels_combined"] = tuple(
                [
                    sum(
                        [
                            backbone["extra"]["stage2"]["num_channels"][i]
                            for backbone in self.inputs_to_combine
                        ]
                    )
                    for i in range(
                        len(
                            self.inputs_to_combine[0]["extra"]["stage2"]["num_channels"]
                        )
                    )
                ]
            )
            return
        elif self.divide_stage == 3:
            self.extra["stage3"]["num_channels_combined"] = tuple(
                [
                    sum(
                        [
                            backbone["extra"]["stage3"]["num_channels"][i]
                            for backbone in self.inputs_to_combine
                        ]
                    )
                    for i in range(
                        len(
                            self.inputs_to_combine[0]["extra"]["stage3"]["num_channels"]
                        )
                    )
                ]
            )
            return
        else:
            raise ValueError("divide stage should be 1, 2, or 3")

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def decide_which_channels_to_use(self):
        if not self.run_part_1 and not self.run_part_2:
            raise ValueError(
                "At least one of 'run_part_1' and 'run_part_2' must be true!"
            )
        if self.divide_stage < 1 or self.divide_stage > 3:
            raise ValueError("divide_after_stage must be 1, 2 or 3")

        if self.run_part_1:
            for i in range(self.divide_stage):
                # If divide_stage is 2, then we need stage 1 and stage 2
                setattr(self, f"build_stage_{i + 1}", True)

        if self.run_part_2:
            for i in range(self.divide_stage, 4):
                setattr(self, f"build_stage_{i + 1}", True)

    def init_weights(self, pretrained=None, override_prefix=""):
        """Initialize the weights in backbone.

        Args:
            override_prefix (str, optional): Expected prefix for the weights
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self,
                pretrained,
                strict=False,
                logger=logger,
                override_prefix=override_prefix,
            )
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x, part: int = 1):
        if part != 1 and part != 2:
            raise ValueError("Part should be 1 or 2")

        run_stage_1 = part == 1
        run_stage_2 = (part == 1 and self.divide_stage > 1) or (
            part == 2 and self.divide_stage < 2
        )
        run_stage_3 = (part == 1 and self.divide_stage > 2) or (
            part == 2 and self.divide_stage < 3
        )
        run_stage_4 = part == 2

        if run_stage_1:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.norm2(x)
            x = self.relu(x)
            x = self.layer1(x)

        if run_stage_2:
            x_list = []
            for i in range(self.stage2_cfg["num_branches"]):
                if self.transition1[i] is not None:
                    x_list.append(self.transition1[i](x))
                else:
                    x_list.append(x)
            x = self.stage2(x_list)

        if run_stage_3:
            x_list = []
            for i in range(self.stage3_cfg["num_branches"]):
                if self.transition2[i] is not None:
                    x_list.append(self.transition2[i](x[-1]))
                else:
                    x_list.append(x[i])
            x = self.stage3(x_list)

        if run_stage_4:
            x_list = []
            for i in range(self.stage4_cfg["num_branches"]):
                if self.transition3[i] is not None:
                    x_list.append(self.transition3[i](x[-1]))
                else:
                    x_list.append(x[i])
            x = self.stage4(x_list)

        return x

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return

        if self.build_stage_1:
            self.norm1.eval()
            self.norm2.eval()
            for m in [self.conv1, self.norm1, self.conv2, self.norm2]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if not getattr(self, f"build_stage_{i}"):
                continue
            lookup = "layer1" if i == 1 else f"stage{i}"
            m = getattr(self, lookup)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i < 4:
                m = getattr(self, f"transition{i}")
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _make_transition_layer(
        self,
        num_channels_pre_layer,
        num_channels_cur_layer,
        num_combined_channels_pre_layer=None,
    ):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)  # 3
        num_branches_pre = len(num_channels_pre_layer)  # 2
        if num_combined_channels_pre_layer is None:
            num_combined_channels_pre_layer = [n for n in num_channels_pre_layer]

        transition_layers = []
        transition_channels = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                # i = 0, i = 1
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    # we should not build a transition layer here even though the numbers
                    # of channels is higher because there are from the same 'branch' - i don't
                    # know the correct terminology here!
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_combined_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])[
                                1
                            ],
                            nn.ReLU(inplace=True),
                        )
                    )
                    transition_channels.append(num_channels_cur_layer[i])
                else:
                    transition_layers.append(None)
                    transition_channels.append(num_combined_channels_pre_layer[i])
            else:
                # i = 2
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    # j = 0
                    in_channels = num_combined_channels_pre_layer[-1]
                    out_channels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else in_channels
                    )
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv_downsamples))
                transition_channels.append(num_channels_cur_layer[i])

        return nn.ModuleList(transition_layers), transition_channels

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        """Make stage."""
        num_modules = layer_config["num_modules"]  # 4
        num_branches = layer_config["num_branches"]  # 3
        num_blocks = layer_config["num_blocks"]  #  (4, 4, 4)
        num_channels = layer_config["num_channels"]  #  (32, 64, 128)
        block = HRNet.blocks_dict[layer_config["block"]]  #  basic

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule(
                    num_branches,  # 3
                    block,  #  basics
                    num_blocks,  #  (4, 4, 4)
                    in_channels,  #  (
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    upsample_cfg=self.upsample_cfg,
                )
            )

            in_channels = hr_modules[-1].in_channels

        return nn.Sequential(*hr_modules), in_channels

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """Make layer."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, out_channels)[1],
            )

        layers = [
            block(
                in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
            )
        ]
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                )
            )

        return nn.Sequential(*layers)

    def prep_inputs(self, inputs: Union[List[Tensor], List[List[Tensor]]]):
        if not self.run_part_2:
            raise ValueError("This BreakableHRNet is not configured for part 2")
        if self.divide_stage == 1:
            # Inputs should just be a list of tensors
            if not all([isinstance(x, Tensor) for x in inputs]):
                raise ValueError(
                    "When dividing after stage 1, inputs should be a list of Tensors"
                )
            new_input = torch.cat(inputs, dim=1)
            return new_input
        elif self.divide_stage == 2:
            # Inputs should be a list of lists of tensors
            if not all(
                [
                    isinstance(x, List) and all([isinstance(y, Tensor) for y in x])
                    for x in inputs
                ]
            ):
                raise ValueError(
                    "When dividing after stage 2, inputs should be a list of lists of Tensors"
                )
            new_input = [
                torch.cat([arr[i] for arr in inputs], dim=1)
                for i in range(len(inputs[0]))
            ]
            return new_input
        elif self.divide_stage == 3:
            # Inputs should be a list of lists of tensors
            if not all(
                [
                    isinstance(x, List) and all([isinstance(y, Tensor) for y in x])
                    for x in inputs
                ]
            ):
                raise ValueError(
                    "When dividing after stage 3, inputs should be a list of lists of Tensors"
                )
            new_input = [
                torch.cat([arr[i] for arr in inputs], dim=1)
                for i in range(len(inputs[0]))
            ]
            return new_input
        raise NotImplementedError()

    @staticmethod
    def prep_output(output) -> Tensor:
        return output[0]

    def num_output_channels(self) -> int:
        return self.stage4_cfg["num_channels"][0]

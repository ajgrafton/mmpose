# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from mmcv import Config

from mmpose.datasets.dataset_info import DatasetInfo


def get_eye_keypoint_ids(model_cfg: Config) -> Tuple[int, int]:
    """A helpfer function to get the keypoint indices of left and right eyes
    from the model config.

    Args:
        model_cfg (Config): pose model config.

    Returns:
        int: left eye keypoint index.
        int: right eye keypoint index.
    """
    left_eye_idx = None
    right_eye_idx = None

    # try obtaining eye point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)
        left_eye_idx = dataset_info.keypoint_name2id.get('left_eye', None)
        right_eye_idx = dataset_info.keypoint_name2id.get('right_eye', None)
    except AttributeError:
        left_eye_idx = None
        right_eye_idx = None

    if left_eye_idx is None or right_eye_idx is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {
                'TopDownCocoDataset', 'TopDownCocoWholeBodyDataset'
        }:
            left_eye_idx = 1
            right_eye_idx = 2
        elif dataset_name in {'AnimalPoseDataset', 'AnimalAP10KDataset'}:
            left_eye_idx = 0
            right_eye_idx = 1
        else:
            raise ValueError('Can not determine the eye keypoint id of '
                             f'{dataset_name}')

    return left_eye_idx, right_eye_idx


def get_face_keypoint_ids(model_cfg: Config) -> Tuple[int, int]:
    """A helpfer function to get the keypoint indices of the face from the
    model config.

    Args:
        model_cfg (Config): pose model config.

    Returns:
        list[int]: face keypoint index.
    """
    face_indices = None

    # try obtaining nose point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)
        for id in range(68):
            face_indices.append(
                dataset_info.keypoint_name2id.get(f'face_{id}', None))
    except AttributeError:
        face_indices = None

    if face_indices is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {'TopDownCocoWholeBodyDataset'}:
            face_indices = list(range(23, 91))
        else:
            raise ValueError('Can not determine the face id of '
                             f'{dataset_name}')

    return face_indices


def get_wrist_keypoint_ids(model_cfg: Config) -> Tuple[int, int]:
    """A helpfer function to get the keypoint indices of left and right wrist
    from the model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        int: left wrist keypoint index.
        int: right wrist keypoint index.
    """

    # try obtaining eye point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)
        left_wrist_idx = dataset_info.keypoint_name2id.get('left_wrist', None)
        right_wrist_idx = dataset_info.keypoint_name2id.get(
            'right_wrist', None)
    except AttributeError:
        left_wrist_idx = None
        right_wrist_idx = None

    if left_wrist_idx is None or right_wrist_idx is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {
                'TopDownCocoDataset', 'TopDownCocoWholeBodyDataset'
        }:
            left_wrist_idx = 9
            right_wrist_idx = 10
        elif dataset_name == 'AnimalPoseDataset':
            left_wrist_idx = 16
            right_wrist_idx = 17
        elif dataset_name == 'AnimalAP10KDataset':
            left_wrist_idx = 7
            right_wrist_idx = 10
        else:
            raise ValueError('Can not determine the eye keypoint id of '
                             f'{dataset_name}')

    return left_wrist_idx, right_wrist_idx


def get_mouth_keypoint_ids(model_cfg: Config) -> Tuple[int, int]:
    """A helpfer function to get the keypoint indices of the left and right
    part of mouth from the model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        int: left-part mouth keypoint index.
        int: right-part mouth keypoint index.
    """
    # try obtaining mouth point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)
        left_mouth_idx = dataset_info.keypoint_name2id.get('face-0', None)
        right_mouth_idx = dataset_info.keypoint_name2id.get('face-16', None)
    except AttributeError:
        left_mouth_idx = None
        right_mouth_idx = None

    if left_mouth_idx is None or right_mouth_idx is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name == 'TopDownCocoWholeBodyDataset':
            left_mouth_idx = 23
            right_mouth_idx = 39
        else:
            raise ValueError('Can not determine the eye keypoint id of '
                             f'{dataset_name}')

    return left_mouth_idx, right_mouth_idx


def get_hand_keypoint_ids(model_cfg: Config) -> List[int]:
    """A helpfer function to get the keypoint indices of left and right hand
    from the model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        list[int]: hand keypoint indices.
    """
    # try obtaining hand keypoint ids from dataset_info
    try:
        hand_indices = []
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)

        hand_indices.append(
            dataset_info.keypoint_name2id.get('left_hand_root', None))

        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_thumb{id}', None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_forefinger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_middle_finger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_ring_finger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_pinky_finger{id}',
                                                  None))

        hand_indices.append(
            dataset_info.keypoint_name2id.get('right_hand_root', None))

        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_thumb{id}', None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_forefinger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_middle_finger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_ring_finger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_pinky_finger{id}',
                                                  None))

    except AttributeError:
        hand_indices = None

    if hand_indices is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {'TopDownCocoWholeBodyDataset'}:
            hand_indices = list(range(91, 133))
        else:
            raise ValueError('Can not determine the hand id of '
                             f'{dataset_name}')

    return hand_indices


def get_hand_heart_keypoint_ids(model_cfg: Config) -> List[int]:
    """A helpfer function to get the related keypoint indices of hand heart
    from the model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        list[int]: the related keypoint indices of hand heart
    """
    # try obtaining hand keypoint ids from dataset_info
    try:
        hand_heart_indices = []
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)

        hand_heart_indices.append(
            dataset_info.keypoint_name2id.get('left_hand_root', None))
        for id in [1, 3, 4]:
            hand_heart_indices.append(
                dataset_info.keypoint_name2id.get(f'left_pinky_finger{id}',
                                                  None))

        hand_heart_indices.append(
            dataset_info.keypoint_name2id.get('right_hand_root', None))

        for id in [1, 3, 4]:
            hand_heart_indices.append(
                dataset_info.keypoint_name2id.get(f'right_pinky_finger{id}',
                                                  None))

    except AttributeError:
        hand_heart_indices = None

    if hand_heart_indices is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {'TopDownCocoWholeBodyDataset'}:
            hand_heart_indices = [91, 108, 110, 111, 112, 129, 131, 132]
        else:
            raise ValueError('Can not determine the hand id of '
                             f'{dataset_name}')

    return hand_heart_indices


def get_flying_heart_keypoint_ids(model_cfg: Config) -> List[int]:
    """A helpfer function to get the related keypoint indices of flying heart
    (blow kiss) from the model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        list[int]: the related keypoint indices of flying heart
    """
    # try obtaining hand keypoint ids from dataset_info
    try:
        flying_heart_indices = []
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)

        for id in [1, 4]:
            flying_heart_indices.append(
                dataset_info.keypoint_name2id.get(f'left_middle_finger{id}',
                                                  None))

        for id in [1, 4]:
            flying_heart_indices.append(
                dataset_info.keypoint_name2id.get(f'right_middle_finger{id}',
                                                  None))

    except AttributeError:
        flying_heart_indices = None

    if flying_heart_indices is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {'TopDownCocoWholeBodyDataset'}:
            flying_heart_indices = [100, 103, 121, 124]
        else:
            raise ValueError('Can not determine the hand id of '
                             f'{dataset_name}')

    return flying_heart_indices

import os.path
import mmcv
import numpy as np
import pickle
import struct
import cv2
from functools import lru_cache
from mmpose.datasets import PIPELINES
from meerkat.training_data import MeerkatDataLoader
from typing import List


@PIPELINES.register_module()
class LoadMultipleImagesFromMeerkat:
    def __init__(
        self,
        to_float32=False,
        color_channel_order="rgb",
        color_start_indices=None,
        file_client_args=None,
    ):
        if color_start_indices is None:
            color_start_indices = []
        if isinstance(color_start_indices, int):
            color_start_indices = [color_start_indices]

        if file_client_args is None:
            file_client_args = dict(backend="disk")
        self.to_float32 = to_float32
        self.color_channel_order = color_channel_order
        self.color_start_indices = color_start_indices
        self._reader = MeerkatDataLoader()

    def __call__(self, results):
        image_file = results.get("image_file", None)
        if isinstance(image_file, (list, tuple)):
            results["img"] = [self._read_image(path) for path in image_file]
        elif image_file is not None:
            results["img"] = self._read_image(image_file)
        else:
            if "img" not in results:
                raise KeyError("Either `image_file` or `img` should exist in results.")
            if isinstance(results["img"], (list, tuple)):
                assert isinstance(results["img"][0], np.ndarray)
            else:
                assert isinstance(results["img"], np.ndarray)

            if self.color_channel_order == "rgb":
                if isinstance(results["img"], (list, tuple)):
                    results["img"] = [self._bgr2rgb(img) for img in results["img"]]
                else:
                    results["img"] = self._bgr2rgb(results["img"])
            results["image_file"] = None

        return results

    @staticmethod
    def _count_number_of_channels(images: List[np.ndarray]):
        """
        Counts the number of colour channels in the provided list of images
        Args:
            images: List of loaded images
        Returns: The number of colour channels in the image
        """
        total_num_channels = 0
        for img in images:
            if img.ndim == 3:
                total_num_channels += img.shape[2]
            else:
                total_num_channels += 1
        return total_num_channels

    def _bgr2rgb(self, image: np.ndarray) -> np.ndarray:
        image_2 = image.copy()
        for color_start in self.color_start_indices:
            image_2[..., color_start] = image[..., color_start + 2]
            image_2[..., color_start + 2] = image[..., color_start]
        return image_2

    @staticmethod
    def _stack_images(images: List[np.ndarray]) -> np.ndarray:
        num_channels = LoadMultipleImagesFromMeerkat._count_number_of_channels(images)
        image_shape = (images[0].shape[0], images[0].shape[1], num_channels)
        stacked_image = np.zeros(image_shape, dtype=np.uint8)
        idx = 0
        for image in images:
            if image.ndim == 3:
                end_idx = idx + image.shape[2]
                stacked_image[..., idx:end_idx] = image
                idx = end_idx
            else:
                stacked_image[..., idx] = image
                idx += 1
        return stacked_image

    def _read_image(self, path: str):
        individual_files = path.split("::")
        # The dataloader automatically applies a directory prefix to the start of the image file.
        # We need to get the prefix from the first file, and apply it to the others
        source_folder = os.path.dirname(individual_files[0])
        for i in range(1, len(individual_files)):
            individual_files[i] = os.path.join(
                source_folder, os.path.basename(individual_files[i])
            )

        images = [
            self._reader.load_image(
                file_name,
                flag="",
                channel_order="bgr",  # Request images as bgr because we need to shuffle them afterwards
            )
            for file_name in individual_files
        ]
        image = self._stack_images(images)
        if self.color_channel_order == "rgb":
            image = self._bgr2rgb(image)
        return image


@PIPELINES.register_module()
class LoadImageFromMeerkat:
    """Loading image(s) from Meerkat file.

    Required key: "image_file".

    Added key: "img".

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(
        self,
        to_float32=False,
        color_type="color",
        channel_order="rgb",
        file_client_args=None,
    ):
        if file_client_args is None:
            file_client_args = dict(backend="disk")
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self._reader = MeerkatDataLoader()

    def _read_image(self, path):
        img = self._reader.load_image(
            path, flag=self.color_type, channel_order=self.channel_order
        )
        return img

    @staticmethod
    def _bgr2rgb(img):
        if img.ndim == 3:
            if img.shape[2] == 5:
                img2 = img.copy()
                img2[:, :, :3] = mmcv.bgr2rgb(img2[:, :, :3])
                return img2
            return mmcv.bgr2rgb(img)
        elif img.ndim == 4:
            return np.concatenate([mmcv.bgr2rgb(img_) for img_ in img], axis=0)
        else:
            raise ValueError('results["img"] has invalid shape ' f"{img.shape}")

    def __call__(self, results):
        """Loading image(s) from file."""
        # if self.file_client is None:
        #     self.file_client = mmcv.FileClient(**self.file_client_args)

        image_file = results.get("image_file", None)

        if isinstance(image_file, (list, tuple)):
            # Load images from a list of paths
            results["img"] = [self._read_image(path) for path in image_file]
        elif image_file is not None:
            # Load single image from path
            results["img"] = self._read_image(image_file)
        else:
            if "img" not in results:
                # If `image_file`` is not in results, check the `img` exists
                # and format the image. This for compatibility when the image
                # is manually set outside the pipeline.
                raise KeyError(
                    "Either `image_file` or `img` should exist in " "results."
                )
            if isinstance(results["img"], (list, tuple)):
                assert isinstance(results["img"][0], np.ndarray)
            else:
                assert isinstance(results["img"], np.ndarray)
            if self.color_type == "color" and self.channel_order == "rgb":
                # The original results['img'] is assumed to be image(s) in BGR
                # order, so we convert the color according to the arguments.
                if isinstance(results["img"], (list, tuple)):
                    results["img"] = [self._bgr2rgb(img) for img in results["img"]]
                else:
                    results["img"] = self._bgr2rgb(results["img"])
            results["image_file"] = None

        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}'"
        )
        return repr_str

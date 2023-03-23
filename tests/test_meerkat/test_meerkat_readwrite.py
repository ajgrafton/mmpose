import numpy as np
import pytest

import cv2
import mmcv
import os

from meerkat.training_data import MeerkatDataLoader, MeerkatDataWriter


def test_roundtrip():
    # generate three temporary files
    writer_1 = MeerkatDataWriter('archive_1a.mka')
    writer_2 = MeerkatDataWriter('archive_2a.mka')
    writer_3 = MeerkatDataWriter('archive_3a.mka')
    writers = [writer_1, writer_2, writer_3]

    images = [
        [np.random.uniform(low=0, high=255, size=(35, 50, 3)).astype(np.uint8) for _ in range(5)]
        for _ in range(3)
    ]
    load_names = [
        [f"archive_{i+1}a.mka:w{i}i{j}.jpg:15_25_5_5" for j in range(5)]
        for i in range(3)
    ]

    for i, writer in enumerate(writers):
        for j, img in enumerate(images[i]):
            writer.add_image(img, f"w{i}i{j}.jpg")
        writer.write()
    
    jpeg_images = [
        [mmcv.imfrombytes(cv2.imencode('.jpg', img)[1]) for img in file_images]
        for file_images in images
    ]

    loader = MeerkatDataLoader()

    for i in range(len(writers)):
        for j in range(len(images[i])):
            loaded_image = loader.load_image(load_names[i][j])
            test_image = jpeg_images[i][j][20:31, 10:21, :]
            np.testing.assert_equal(loaded_image, test_image)
    os.remove('archive_1a.mka')
    os.remove('archive_2a.mka')
    os.remove('archive_3a.mka')


def test_roundtrip_2():
    # generate three temporary files
    writer_1 = MeerkatDataWriter('archive_1b.mka')
    writer_2 = MeerkatDataWriter('archive_2b.mka')
    writer_3 = MeerkatDataWriter('archive_3b.mka')
    writers = [writer_1, writer_2, writer_3]

    images = [
        [np.random.uniform(low=0, high=255, size=(10, 10, 3)).astype(np.uint8) for _ in range(5)]
        for _ in range(3)
    ]
    load_names = [
        [f"archive_{i+1}b.mka:w{i}i{j}.jpg:5_5_10_10" for j in range(5)]
        for i in range(3)
    ]

    for i, writer in enumerate(writers):
        for j, img in enumerate(images[i]):
            writer.add_image(img, f"w{i}i{j}.jpg")
        writer.write()
    
    jpeg_images = [
        [mmcv.imfrombytes(cv2.imencode('.jpg', img)[1]) for img in file_images]
        for file_images in images
    ]

    loader = MeerkatDataLoader()

    for i in range(len(writers)):
        for j in range(len(images[i])):
            loaded_image = loader.load_image(load_names[i][j])
            test_image = jpeg_images[i][j]
            print(loaded_image[:, :, 0])
            np.testing.assert_equal(loaded_image[5:15, 5:15, :], test_image)
    

    os.remove('archive_1b.mka')
    os.remove('archive_2b.mka')
    os.remove('archive_3b.mka')


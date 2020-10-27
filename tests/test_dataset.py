import unittest
import gtsam
import numpy as np
from visual_gtsam.dataset import Dataset
from visual_gtsam.dataset.structures import Image, Imu, Range, Transformation, Vector3D
from datetime import datetime, timedelta
import random
import cv2
import os


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = Dataset()
        self.assertIsInstance(dataset.get_next_image(), Image)
        self.assertIsInstance(dataset.get_next_imu(), Imu)
        for i in range(random.randint(0, 100)):
            dataset.get_next_image().get_origin_rgb()
        (image, imu) = dataset.get_next_pair()
        print("Delta between image and imu is (seconds)", image.get_time() - imu.get_time())
        self.assertAlmostEqual(image.get_time().get_time(), imu.get_time().get_time(), delta=timedelta(milliseconds=10))
        print(dataset.get_statistic())

    def test_dataset_new_structures(self):
        dataset = Dataset()
        self.assertIsInstance(dataset.get_next_range(), Range)
        self.assertIsInstance(dataset.get_next_ur(), Transformation)
        range = dataset.get_next_range()
        print(range)
        self.assertIsInstance(range.get_value(), float)
        transform = dataset.get_next_ur()
        print(transform)
        self.assertIsInstance(transform.get_rotation(), np.ndarray)


    def test_collect_images_for_dataset(self):
        dataset = Dataset()
        save_frame = 0
        save_every = 6
        path_dict_to_save = "./downloads/saved"
        if not os.path.exists(path_dict_to_save):
            os.makedirs(path_dict_to_save, exist_ok=True)

        for image in dataset.get_image_sequence():
            image_for_save = image.get_origin_rgb()
            # cv2.imshow("image", image_for_save)
            # cv2.waitKey()
            image_name = image.get_file_name()
            if save_frame == 0:
                path_to_save = "{}/{}".format(path_dict_to_save, image_name)
                cv2.imwrite(path_to_save, image_for_save)
            save_frame = (save_frame + 1) % save_every

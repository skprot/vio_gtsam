import datetime
import unittest
import numpy as np
import cv2
from visual_gtsam.dataset.structures import Timestamp, Vector3D, Imu, ImuSequence, ImageSequence, Image, RangeSequence, \
    URSequence


class TestStructures(unittest.TestCase):
    _data_imu_test = {'time': {'secs': 1594292635, 'nsecs': 87993127},
                      'angular_velocity': {'x': -6.877862453460693, 'y': 3.7938930988311768, 'z': 2.435114622116089},
                      'linear_acceleration': {'x': 0.9289833903312683, 'y': 0.14844579994678497,
                                              'z': 8.293810844421387}}
    _data_image_test = {"path": "images/1594379846_0111020088.png", "time": {"secs": 1594379846, "nsecs": 111020088}}

    _data_range_test = None
    _data_ur_test = None

    def test_time(self):
        secs = self._data_imu_test["time"]["secs"]
        nsecs = self._data_imu_test["time"]["nsecs"]
        time = Timestamp(secs, nsecs)
        self.assertIsInstance(time.get_time_str(), str)
        self.assertIsInstance(time.get_time(), datetime.datetime)

        time1 = Timestamp(secs + 1, nsecs)
        time2 = Timestamp(secs, nsecs + 1000)
        print(time, time1, time2)
        self.assertTrue(time1 > time)
        self.assertTrue(time2 > time)
        self.assertTrue(time1 > time2)

    def test_vector3d(self):
        accel_origin = self._data_imu_test["linear_acceleration"]
        acceleration = Vector3D(accel_origin["x"], accel_origin["y"], accel_origin["z"])
        print(acceleration)
        self.assertIsInstance(acceleration.get_vector(), np.ndarray)

    def test_imu(self):
        t_dict = self._data_imu_test["time"]
        a_dict = self._data_imu_test["angular_velocity"]
        v_dict = self._data_imu_test["linear_acceleration"]
        imu_data = Imu(t_dict["secs"], t_dict["nsecs"],
                       a_dict["x"], a_dict["y"], a_dict["z"],
                       v_dict["x"], v_dict["y"], v_dict["z"])

        print(imu_data)
        self.assertIsInstance(imu_data.get_time(), Timestamp)
        self.assertIsInstance(imu_data.get_velocity(), np.ndarray)
        self.assertIsInstance(imu_data.get_acceleration(), np.ndarray)

    def test_image(self):
        t_dict = self._data_image_test["time"]
        path = self._data_image_test["path"]
        try:
            camera_matrix = np.load("downloads/calibration.npy", allow_pickle=True, encoding='latin1')[0]
            image = Image(t_dict["secs"], t_dict["nsecs"], "downloads", path, camera_matrix)
            self.assertIsInstance(image.get_origin_rgb(), np.ndarray)
            self.assertIsInstance(image.get_origin_rgb().shape, tuple)
            self.assertEqual(image.get_origin_rgb().shape, (480, 640, 3))
            image_origin = image.get_origin_rgb()
            image_undist = image.get_undistorted()
            cv2.imshow("origin", image_origin)
            cv2.imshow("undist", image_undist)
            cv2.waitKey(0)
        except FileNotFoundError:
            camera_matrix = None
            image = Image(t_dict["secs"], t_dict["nsecs"], "downloads", path, camera_matrix)
            self.assertIsInstance(image.get_origin_rgb(), np.ndarray)
            self.assertIsInstance(image.get_origin_rgb().shape, tuple)
            self.assertEqual(image.get_origin_rgb().shape, (480, 640, 3))

    def test_imu_sequence(self):
        print("---Start testing of IMU sequence")
        imu_seq = ImuSequence()
        self.assertIsInstance(imu_seq.get_length(), int)
        self.assertTrue(imu_seq.get_length() > 0)
        self.assertEqual(imu_seq.get_all_acceleration().shape, (imu_seq.get_length(), 3))
        self.assertEqual(imu_seq.get_all_velocity().shape, (imu_seq.get_length(), 3))
        self.assertIsInstance(imu_seq.get_next(), Imu)
        print("---Stop testing of IMU sequence")

    def test_image_sequence(self):
        print("---Start testing of IMAGE sequence")
        image_seq = ImageSequence()
        self.assertIsInstance(image_seq.get_length(), int)
        self.assertTrue(image_seq.get_length() > 0)
        self.assertIsInstance(image_seq.get_next(), Image)
        self.assertIsInstance(image_seq.get_next().get_origin_rgb(), np.ndarray)
        self.assertIsInstance(image_seq.get_next().get_origin_rgb().shape, tuple)
        self.assertEqual(image_seq.get_next().get_origin_rgb().shape, (480, 640, 3))
        print("---Stop testing of IMAGE sequence")

    def test_range_sequence(self):
        seq = RangeSequence()
        ranges = seq.get_all_ranges()
        self.assertIsInstance(ranges, np.ndarray)
        self.assertEqual(len(ranges), seq.get_length())
        self.assertAlmostEqual(1.0, ranges.mean(), delta=0.05)

    def test_ur_sequence(self):
        seq = URSequence()
        true_trajectory = seq.get_true_trajectory()
        self.assertIsInstance(true_trajectory, np.ndarray)
        self.assertEqual(len(true_trajectory), seq.get_length())
        print(true_trajectory[0])

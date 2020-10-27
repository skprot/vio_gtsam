import cv2
import numpy as np
from visual_gtsam.dataset.structures import Timestamp


class Image(object):
    def __init__(self, t_secs: int, t_nsecs: int, main_dir: str, path_to_img: str):
        self._time = Timestamp(t_secs, t_nsecs)
        self._path_to_img = path_to_img
        self._main_dir = main_dir
        # self._calibration_matrix = calibration_matrix

    def __repr__(self):
        return "Time: {}: path: {}".format(self._time.get_time_str(), self._get_path())

    def get_time(self) -> Timestamp:
        return self._time

    def _get_path(self):
        return "{}/{}".format(self._main_dir, self._path_to_img)

    def get_file_name(self):
        return self._path_to_img.split("/")[-1]

    def get_origin_rgb(self):
        return cv2.imread(self._get_path())

    # def get_undistorted(self):
    #     image = cv2.imread(self._get_path())
    #     k = self._calibration_matrix["k"]
    #     d = self._calibration_matrix["d"]
    #     dim = self._calibration_matrix["dim"]
    #     map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dim, cv2.CV_16SC2)
    #     return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

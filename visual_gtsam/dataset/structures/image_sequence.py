import json
import numpy as np
from visual_gtsam.dataset.structures.image import Image


class ImageSequence(object):
    _calibration_matrix: np.ndarray

    def __init__(self, main_dir='downloads', image_dir='images', image_json_filename='image_data.json',
                 calibration_path="calibration.npy"):
        self._main_dir = main_dir
        self._image_dir = image_dir
        self._image_json_filename = image_json_filename
        self._length = 0
        self._current_idx = 0
        self._calibration_matrix_path = calibration_path
        # self._load_calibration_matrix()
        self._image_data = []
        self._load_data()

    def _load_calibration_matrix(self) -> None:
        self._calibration_matrix = \
            np.load("{}/{}".format(self._main_dir, self._calibration_matrix_path), allow_pickle=True,
                    encoding='latin1')[0]

    def _load_data(self):
        with open("{0}/{1}".format(self._main_dir, self._image_json_filename), "r") as read_file:
            json_data = json.load(read_file)
            for image_json in json_data:
                try:
                    t_dict = image_json["time"]
                    path = image_json["path"]
                    self._image_data.append(
                        Image(t_dict["secs"], t_dict["nsecs"], self._main_dir, path))
                except KeyError:
                    continue
            self._length = len(self._image_data)

    def get_length(self) -> int:
        return self._length

    def get_next(self) -> Image:
        return next(self)

    def reset(self):
        self._current_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Image:
        if self._current_idx < self._length:
            self._current_idx += 1
            return self._image_data[self._current_idx - 1]
        else:
            self._current_idx = 0
            raise StopIteration

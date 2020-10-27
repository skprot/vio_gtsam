import json
import numpy as np

from visual_gtsam.dataset.structures import Imu


class ImuSequence(object):
    def __init__(self, main_dir='downloads', imu_json_filename='imus_data.json'):
        self._main_dir = main_dir
        self._imu_json_filename = imu_json_filename
        self._length = 0
        self._current_idx = 0
        self._imu_data = []
        self._load_data()

    def _load_data(self):
        with open("{0}/{1}".format(self._main_dir, self._imu_json_filename), "r") as read_file:
            json_data = json.load(read_file)
            for imu_json in json_data:
                try:
                    t_dict = imu_json["time"]
                    a_dict = imu_json["angular_velocity"]
                    v_dict = imu_json["linear_acceleration"]
                    self._imu_data.append(Imu(t_dict["secs"], t_dict["nsecs"],
                                              a_dict["x"], a_dict["y"], a_dict["z"],
                                              v_dict["x"], v_dict["y"], v_dict["z"]))
                except KeyError:
                    continue
            self._length = len(self._imu_data)

    def get_length(self) -> int:
        return self._length

    def get_next(self) -> Imu:
        return next(self)

    def get_all_acceleration(self) -> np.array:
        values = None
        for imu in self:
            if values is not None:
                values = np.vstack((values, imu.get_acceleration()))
            else:
                values = imu.get_acceleration()
        return values

    def get_all_velocity(self) -> np.array:
        values = None
        for imu in self:
            if values is not None:
                values = np.vstack((values, imu.get_velocity()))
            else:
                values = imu.get_velocity()
        return values

    def reset(self):
        self._current_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Imu:
        if self._current_idx < self._length:
            self._current_idx += 1
            return self._imu_data[self._current_idx - 1]
        else:
            self._current_idx = 0
            raise StopIteration

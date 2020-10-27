import numpy as np

from visual_gtsam.dataset.structures import Timestamp, Vector3D


class Transformation(object):
    def __init__(self, t_secs, t_nsecs, tx, ty, tz, rx, ry, rz):
        self._time = Timestamp(t_secs, t_nsecs)
        self._translation = Vector3D(tx, ty, tz)
        self._rotation = Vector3D(rx, ry, rz)

    def __repr__(self):
        return "Time: {}: translation: {}; rotation: {}".format(self._time.get_time_str(),
                                                                self._translation.get_vector_str(),
                                                                self._rotation.get_vector_str())

    def get_time(self) -> Timestamp:
        return self._time

    def get_translation(self) -> np.ndarray:
        return self._translation.get_vector()

    def get_rotation(self) -> np.ndarray:
        return self._rotation.get_vector()

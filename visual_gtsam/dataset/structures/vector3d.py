import numpy as np


class Vector3D(object):
    def __init__(self, x: float, y: float, z: float):
        self._x = x
        self._y = y
        self._z = z

    def get_vector(self):
        return np.array([self._x, self._y, self._z])

    def get_vector_str(self):
        return "X = {}\tY = {}\t Z = {}".format(self._x, self._y, self._z)

    def __repr__(self):
        return self.get_vector_str()

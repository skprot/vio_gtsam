import json
import numpy as np

from visual_gtsam.dataset.structures import Transformation


class URSequence(object):
    def __init__(self, main_dir='downloads', json_filename='ur_data.json'):
        self._main_dir = main_dir
        self._json_filename = json_filename
        self._length = 0
        self._current_idx = 0
        self._data = []
        self._load_data()

    def _load_data(self):
        with open("{0}/{1}".format(self._main_dir, self._json_filename), "r") as read_file:
            json_data = json.load(read_file)
            for json_line in json_data:
                try:
                    time_dict = json_line["time"]
                    tr_dict = json_line["translation"]
                    ro_dict = json_line["rotation"]
                    self._data.append(Transformation(time_dict["secs"], time_dict["nsecs"],
                                                     tr_dict["x"], tr_dict["y"], tr_dict["z"],
                                                     ro_dict["x"], ro_dict["y"], ro_dict["z"]))
                except KeyError:
                    continue
            self._length = len(self._data)

    def get_length(self) -> int:
        return self._length

    def get_next(self) -> Transformation:
        return next(self)

    def reset(self):
        self._current_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Transformation:
        if self._current_idx < self._length:
            self._current_idx += 1
            return self._data[self._current_idx - 1]
        else:
            self._current_idx = 0
            raise StopIteration

    def get_true_trajectory(self) -> np.array:
        values = None
        for transform in self:
            if values is not None:
                values = np.vstack((values, transform.get_translation()))
            else:
                values = transform.get_translation()
        return values

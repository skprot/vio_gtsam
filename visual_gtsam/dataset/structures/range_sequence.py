import json

import numpy as np

from visual_gtsam.dataset.structures import Range


class RangeSequence(object):
    def __init__(self, main_dir='downloads', range_json_filename='range_data.json'):
        self._main_dir = main_dir
        self._json_filename = range_json_filename
        self._length = 0
        self._current_idx = 0
        self._data = []
        self._load_data()

    def _load_data(self):
        with open("{0}/{1}".format(self._main_dir, self._json_filename), "r") as read_file:
            json_data = json.load(read_file)
            for json_line in json_data:
                try:
                    t_dict = json_line["time"]
                    range_value = json_line["range"]
                    self._data.append(Range(t_dict["secs"], t_dict["nsecs"], range_value))
                except KeyError as e:
                    print(f"Couldn't find key {e} in source json file {self._json_filename}")
                    continue
            self._length = len(self._data)

    def get_length(self) -> int:
        return self._length

    def get_next(self) -> Range:
        return next(self)

    def reset(self):
        self._current_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Range:
        if self._current_idx < self._length:
            self._current_idx += 1
            return self._data[self._current_idx - 1]
        else:
            self._current_idx = 0
            raise StopIteration

    def get_all_ranges(self) -> np.array:
        values = []
        for range in self:
            values.append(range.get_value())
        return np.asarray(values)

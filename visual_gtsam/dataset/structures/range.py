from visual_gtsam.dataset.structures import Timestamp


class Range(object):
    def __init__(self, t_secs, t_nsecs, value):
        self._time = Timestamp(t_secs, t_nsecs)
        self._value = float(value)

    def __repr__(self):
        return "Time: {}: value: {}".format(self._time.get_time_str(),
                                            self._value)

    def get_time(self) -> Timestamp:
        return self._time

    def get_value(self) -> float:
        return self._value

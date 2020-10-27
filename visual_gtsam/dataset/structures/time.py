from datetime import datetime


class Timestamp:
    def __init__(self, secs: int, nsecs: int):
        self._secs = secs
        self._nsecs = nsecs
        self._time = datetime.fromtimestamp(float("{0}.{1:09d}".format(secs, nsecs)))

    def get_time_str(self):
        return self._time.isoformat()

    def get_time(self):
        return self._time

    def __lt__(self, other):
        return self.get_time() < other.get_time()

    def __le__(self, other):
        return self.get_time() <= other.get_time()

    def __gt__(self, other):
        return self.get_time() > other.get_time()

    def __ge__(self, other):
        return self.get_time() >= other.get_time()

    def __repr__(self):
        return self.get_time_str()

    def __sub__(self, other):
        return self.get_time() - other.get_time()

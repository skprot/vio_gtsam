from visual_gtsam.dataset.structures import Timestamp, Vector3D


class Imu(object):
    def __init__(self, t_secs, t_nsecs, ax, ay, az, vx, vy, vz):
        self._time = Timestamp(t_secs, t_nsecs)
        self._acceleration = Vector3D(ax, ay, az)
        self._velocity = Vector3D(vx, vy, vz)

    def __repr__(self):
        return "Time: {}: acceleration: {}; velocity: {}".format(self._time.get_time_str(),
                                                                 self._acceleration.get_vector_str(),
                                                                 self._velocity.get_vector_str())

    def get_time(self) -> Timestamp:
        return self._time

    def get_acceleration(self):
        return self._acceleration.get_vector()

    def get_velocity(self):
        return self._velocity.get_vector()

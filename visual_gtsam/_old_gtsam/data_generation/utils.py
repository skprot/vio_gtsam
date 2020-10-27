"""
Sudhanva Sreesha
ssreesha@umich.edu
21-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018

General utilities available to the filter and internal functions.
"""

import numpy as np
from data_generation.field_map import FieldMap


def wrap_angle(angle):
    """
    Wraps the given angle to the range [-pi, +pi].

    :param angle: The angle (in rad) to wrap (can be unbounded).
    :return: The wrapped angle (guaranteed to in [-pi, +pi]).
    """

    pi2 = 2 * np.pi

    while angle < -np.pi:
        angle += pi2

    while angle >= np.pi:
        angle -= pi2

    return angle


def get_observation(state, field_map, lm_id):
    """
    Generates a sample observation given the current state of the robot and the marker id of which to observe.

    :param state: The current state of the robot (format: [x, y, theta]).
    :param field_map: A map of the field.
    :param lm_id: The landmark id indexing into the landmarks list in the field map.
    :return: The observation to the landmark (format: np.array([range, bearing, landmark_id])).
             The bearing (in rad) will be in [-pi, +pi].
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(field_map, FieldMap)

    assert state.shape == (3,)

    lm_id = int(lm_id)

    dx = field_map.landmarks_poses_x[lm_id] - state[0]
    dy = field_map.landmarks_poses_y[lm_id] - state[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)
    bearing = np.arctan2(dy, dx) - state[2]

    return np.array([distance, wrap_angle(bearing), lm_id])


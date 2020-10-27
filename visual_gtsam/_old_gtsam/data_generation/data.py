"""
Sudhanva Sreesha
ssreesha@umich.edu
21-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018
"""

import os

import numpy as np
from data_generation.objects import SimulationData
from data_generation.objects import SlamDebugData
from data_generation.objects import SlamInputData


def load_data(data_filename):
    """
    Load existing data from a given filename.
    Accepted file formats are pickled `npy` and MATLAB `mat` extensions.

    :param data_filename: The path to the file with the pre-generated data.
    :raises Exception if the file does not exist.
    :return: DataFile type.
    """

    if not os.path.isfile(data_filename):
        raise Exception('The data file {} does not exist'.format(data_filename))

    file_extension = data_filename[-3:]
    if file_extension not in {'mat', 'npy'}:
        raise TypeError('{} is an unrecognized file extension. Accepted file '
                        'formats include "npy" and "mat"'.format(file_extension))

    num_steps = 0
    filter_data = None
    debug_data = None

    if file_extension == 'npy':
        with np.load(data_filename) as data:
            num_steps = np.asscalar(data['num_steps'])
            filter_data = SlamInputData(data['noise_free_motion'], data['real_observations'])
            debug_data = SlamDebugData(data['real_robot_path'],
                                       data['noise_free_robot_path'],
                                       data['noise_free_observations'])
    elif file_extension == 'mat':
        data = scipy.io.loadmat(data_filename)
        if 'data' not in data:
            raise TypeError('Unrecognized data file')

        data = data['data']
        num_steps = data.shape[0]

        # Convert to zero-indexed landmark IDs.
        data[:, 1] -= 1
        data[:, 6] -= 1

        filter_data = SlamInputData(data[:, 2:5], data[:, 0:2])
        debug_data = SlamDebugData(data[:, 7:10], data[:, 10:13], data[:, 5:7])

    return SimulationData(num_steps, filter_data, debug_data)

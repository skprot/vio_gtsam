"""
Sudhanva Sreesha
ssreesha@umich.edu
22-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018
"""

import os
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt

from data_generation.field_map import FieldMap
from data_generation.objects import Gaussian
from data_generation.data import load_data
from data_generation.plot import plot_robot
from data_generation.plot import plot_field
from data_generation.plot import plot_observations

from gtsam_solver import GtsamSolver
from isam_solver import IsamSolver

def get_cli_args():
    parser = ArgumentParser('Perception in Robotics PS3')
    parser.add_argument('-i',
                        '--input-data-file',
                        type=str,
                        action='store',
                        help='File with generated data to simulate the filter '
                             'against. Supported format: "npy", and "mat".')

    parser.add_argument('-a',
                        '--alphas',
                        type=float,
                        nargs=4,
                        metavar=('A1', 'A2', 'A3', 'A4'),
                        action='store',
                        help='Diagonal of Standard deviations of the Transition noise in action space (M_t).',
                        default=(0.05, 0.001, 0.05, 0.01))

    parser.add_argument('-b',
                        '--beta',
                        type=float,
                        nargs=2,
                        metavar=('range', 'bearing (deg)'),
                        action='store',
                        help='Diagonal of Standard deviations of the Observation noise (Q).',
                        default=(10., 10.))
    parser.add_argument('--dt', type=float, action='store', help='Time step (in seconds).', default=0.1)
    parser.add_argument('-s', '--animate', action='store_true', help='Show and animation of the simulation, in real-time.')
    parser.add_argument('--plot-pause-len',
                        type=float,
                        action='store',
                        help='Time (in seconds) to pause the plot animation for between frames.',
                        default=0.01)
    parser.add_argument('--num-landmarks-per-side',
                        type=int,
                        help='The number of landmarks to generate on one side of the field.',
                        default=4)
    parser.add_argument('--max-obs-per-time-step',
                        type=int,
                        help='The maximum number of observations to generate per time step.',
                        default=2)
    parser.add_argument('--data-association',
                        type=str,
                        choices=['known', 'ml', 'jcbb'],
                        default='known',
                        help='The type of data association algorithm to use during the update step.')

    return parser.parse_args()


def validate_cli_args(args):
    if args.input_data_file and not os.path.exists(args.input_data_file):
        raise OSError('The input data file {} does not exist.'.format(args.input_data_file))

    if not args.input_data_file:
        raise RuntimeError('--input-data-file`  were present in the arguments.')


def main():
    args = get_cli_args()
    validate_cli_args(args)
    alphas = np.array(args.alphas)
    beta = np.array(args.beta)

    mean_prior = np.array([180., 50., 0.])
    Sigma_prior = 1e-12 * np.eye(3, 3)
    initial_state = Gaussian(mean_prior, Sigma_prior)

    if args.input_data_file:
        data = load_data(args.input_data_file)
    else:
        raise RuntimeError('')

    should_show_plots = True if args.animate else False
    should_update_plots = True if should_show_plots else False

    field_map = FieldMap(args.num_landmarks_per_side)

    sam_gt = GtsamSolver(mean_prior, Sigma_prior[0, 0], alphas, beta)
    #sam_gt_isam = IsamSolver(mean_prior, Sigma_prior[0, 0], alphas, beta)

    for t in range(data.num_steps):
        # Used as means to include the t-th time-step while plotting.
        tp1 = t + 1

        # Control at the current step.
        u = data.filter.motion_commands[t]
        # Observation at the current step.
        z = data.filter.observations[t]

        # TODO SLAM update
        #sam_gt_isam.update(u, z)
        sam_gt.update(u, z)

        if not should_update_plots:
            continue

        plt.cla()
        plot_field(field_map, z)
        plot_robot(data.debug.real_robot_path[t])
        plot_observations(data.debug.real_robot_path[t],
                          data.debug.noise_free_observations[t],
                          data.filter.observations[t])

        plt.plot(data.debug.real_robot_path[1:tp1, 0], data.debug.real_robot_path[1:tp1, 1], 'm')
        plt.plot(data.debug.noise_free_robot_path[1:tp1, 0], data.debug.noise_free_robot_path[1:tp1, 1], 'g')

        plt.plot([data.debug.real_robot_path[t, 0]], [data.debug.real_robot_path[t, 1]], '*r')
        plt.plot([data.debug.noise_free_robot_path[t, 0]], [data.debug.noise_free_robot_path[t, 1]], '*g')

        # TODO plot SLAM soltion
        # Plot solution - the whole trajectory and map with 3-sigma error ellipses
        try:
            #plt.plot(sam_gt_isam.states_new[0:, 0], sam_gt_isam.states_new[0:, 1], 'b')
            plt.plot(sam_gt.states_new[0:, 0], sam_gt.states_new[0:, 1], 'b')
            #plt.plot(sam_gt_isam.observation_new[0:, 0], sam_gt_isam.observation_new[0:, 1], 'go', markersize=3)
        except IndexError:
            pass

        if should_show_plots:
            # Draw all the plots and pause to create an animation effect.
            plt.draw()
            plt.pause(args.plot_pause_len)

    plt.show(block=True)


if __name__ == '__main__':
    main()

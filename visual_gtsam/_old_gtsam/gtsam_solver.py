import gtsam
import numpy as np
from data_generation.utils import wrap_angle


class GtsamSolver:
    def __init__(self, initial_state, variance, alphas, beta):
        self._initial_state = gtsam.Pose2(initial_state[0], initial_state[1], initial_state[2])
        self._prior_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([variance, variance, variance]))
        self.observation_noise = gtsam.noiseModel_Diagonal.Sigmas(
            np.array(([np.deg2rad(beta[1]) ** 2, (beta[0] / 100) ** 2])))
        self.alphas = alphas ** 2 / 100
        self.pose_num = 0
        self.landmark_indexes = list()
        self.states_new = np.array([[]])

        self.graph = gtsam.NonlinearFactorGraph()
        self.estimations = gtsam.Values()
        self.result = gtsam.Values()

        self.graph.add(gtsam.PriorFactorPose2(self.pose_num, self._initial_state, self._prior_noise))
        self.estimations.insert(self.pose_num, self._initial_state)

    @staticmethod
    def _get_motion_noise_covariance(motion, alphas):

        """
            Motion noise covariance with given alphas
        """

        drot1, dtran, drot2 = motion
        a1, a2, a3, a4 = alphas

        return np.array([a1 * drot1 ** 2 + a2 * dtran ** 2,
                a3 * dtran ** 2 + a4 * (drot1 ** 2 + drot2 ** 2),
                a1 * drot2 ** 2 + a2 * dtran ** 2])

    @staticmethod
    def _get_motion_prediction(state, motion):

        """
            Predicts the next state given state and the motion command.
        """

        x = state.x()
        y = state.y()
        theta = state.theta()

        drot1, dtran, drot2 = motion

        theta += drot1
        x += dtran * np.cos(theta)
        y += dtran * np.sin(theta)
        theta += drot2

        # Wrap the angle between [-pi, +pi].
        theta = wrap_angle(theta)

        return gtsam.Pose2(x, y, theta)

    @staticmethod
    def _get_landmark_position(state, distance, bearing):
        """
            Predicts the landmark position based on a current state and observation distance and bearing.
        """
        angle = wrap_angle(state.theta() + bearing)
        x_relative = distance * np.cos(angle)
        y_relative = distance * np.sin(angle)
        x = x_relative + state.x()
        y = y_relative + state.y()

        return gtsam.Point2(x, y)

    @staticmethod
    def _get_motion_gtsam_format(motion):
        """
            Predicts the landmark position based on a current state and observation distance and bearing.
        """
        drot1, dtran, drot2 = motion

        theta = drot1 + drot2
        x = dtran * np.cos(theta)
        y = dtran * np.sin(theta)

        # Wrap the angle between [-pi, +pi].
        theta = wrap_angle(theta)

        return gtsam.Pose2(x, y, theta)

    def _convert_to_np_format(self):
        """
            Converts from gtsam.Pose2 to numpy format.
        """
        states = list()
        for i in range(self.result.size() - len(self.landmark_indexes)):
            states.append([self.result.atPose2(i).x(), self.result.atPose2(i).y()])

        self.states_new = np.array(states)

    def update(self, motion, measurement):
        odometry = self._get_motion_gtsam_format(motion)
        noise = gtsam.noiseModel_Diagonal.Sigmas(self._get_motion_noise_covariance(motion, self.alphas))
        predicted_state = self._get_motion_prediction(self.estimations.atPose2(self.pose_num), motion)

        self.graph.add(gtsam.BetweenFactorPose2(self.pose_num, self.pose_num + 1, odometry, noise))
        self.estimations.insert(self.pose_num + 1, predicted_state)

        for i in range(len(measurement)):
            bearing = gtsam.Rot2(measurement[i, 1])
            distance = measurement[i, 0]
            landmark_id = 1000 + measurement[i, 2]

            self.graph.add(gtsam.BearingRangeFactor2D(self.pose_num, landmark_id, bearing, distance, self.observation_noise))

            if landmark_id not in self.landmark_indexes:
                self.landmark_indexes.append(landmark_id)
                landmark_position = self._get_landmark_position(self.estimations.atPose2(self.pose_num), distance, bearing.theta())
                self.estimations.insert(landmark_id, landmark_position)
            else:
                landmark_position = self._get_landmark_position(self.estimations.atPose2(self.pose_num), distance, bearing.theta())
                self.estimations.update(landmark_id, landmark_position)

        #params = gtsam.LevenbergMarquardtParams()
        #params = gtsam.NonlinearOptimizerParams()

        #optimiser = gtsam.LevenbergMarquardtOptimizer(self.graph, self.estimations, params)
        #optimiser = gtsam.NonlinearOptimizer(self.graph, self.estimations, params)
        optimiser = gtsam.GaussNewtonOptimizer(self.graph, self.estimations)

        self.result = optimiser.optimize()
        self.estimations = self.result
        self.pose_num += 1
        self._convert_to_np_format()


import unittest
import numpy as np
from gtsam_solver import GtsamSolver


class GraphTest(unittest.TestCase):
    def setUp(self) -> None:

        """
        setup slam class with testing values
        """

        initial_state = np.array([0, 0, 0])
        covariance = 0.2
        alphas = np.array([0.1, 0.01, 0.1, 0.2])
        betas = np.array([1, 1])
        self.slam = GtsamSolver(initial_state, covariance, alphas, betas)

    def tearDown(self) -> None:

        """
        destruct slam class
        """

        del self.slam

    def test_initial_graph_size(self):

        """
        testing initial graph, estimation vector and result vector sizes
        """

        self.assertEqual(self.slam.graph.size(), 1)
        self.assertEqual(self.slam.estimations.size(), 1)
        self.assertEqual(self.slam.result.size(), 0)

    def test_update(self):

        """
        testing updated graph, estimation vector and result vector sizes with testing values
        """

        motion = np.array([0, 1, 1])
        measurement = np.array([[1, 1, 0],
                                [1, 3, 1],
                                [2, 2, 2],
                                [6, 3, 3]])
        self.slam.update(motion, measurement)

        self.assertEqual(self.slam.graph.size(), 6)
        self.assertEqual(self.slam.estimations.size(), 6)
        self.assertEqual(self.slam.result.size(), 6)

        motion = np.array([0, 1, 1])
        measurement = np.array([[1, 1, 4]])

        self.slam.update(motion, measurement)

        self.assertEqual(self.slam.graph.size(), 8)
        self.assertEqual(self.slam.estimations.size(), 8)
        self.assertEqual(self.slam.result.size(), 8)

    def test_non_zero_noise(self):
        """
        testing non-zero observation noise
        """

        matrix = [self.slam.observation_noise.R()[0, 0], self.slam.observation_noise.R()[0, 1],
                  self.slam.observation_noise.R()[1, 0], self.slam.observation_noise.R()[1, 1]]

        self.assertNotEqual(matrix, [0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()

import gtsam
import numpy as np
from visual_gtsam.barcode_detector import BarcodeDetector
from visual_gtsam.dataset import Dataset
from gtsam.symbol_shorthand import B, V, X
from gtsam.utils import plot

def get_imu_params(vec):
    """Create default parameters with Z *up* and realistic noise parameters"""
    params = gtsam.PreintegrationParams([-0.0, -0.0, 0])
    #([-0.00009042461, -5.37165769 / 10e5, -8.20811897])
    #params = gtsam.PreintegrationParams(vec)

    k_gyro_sigma = 4e-06 # 0.5 degree ARW
    k_accel_sigma = 0.0002  # 10 cm VRW (RANDOM WALK)

    params.setGyroscopeCovariance(
        k_gyro_sigma ** 2 * np.identity(3, np.float))
    params.setAccelerometerCovariance(
        k_accel_sigma ** 2 * np.identity(3, np.float))
    params.setIntegrationCovariance(
        0.0000001 ** 2 * np.identity(3, np.float))

    return params

def get_landmark_points(u, v, d, instrict_matrix): # 3D point in camera frame
    f_x, f_y, c_x, c_y = instrict_matrix
    a = (u - c_x) / f_x
    b = (v - c_y) / f_y

    z = d / np.sqrt(a ** 2 + b ** 2 + 1)
    x = z * a
    y = z * b

    camera_point = gtsam.Point3(x, y, z)

    return camera_point


def transform_to_world_frame(T_cam_imu_mat, pose, camera_point): #TODO: [3]

    camera_pose = gtsam.Pose3(T_cam_imu_mat) * pose
    world_point = camera_pose.transform_from(camera_point)

    return world_point


def get_prediction(prev_pose, prev_speed, prev_bias, preintegration):
    optimized_state = gtsam.NavState(prev_pose, prev_speed)
    predicted_state = preintegration.predict(optimized_state, prev_bias)

    return predicted_state.pose(), predicted_state.velocity()


def run():
    dataset = Dataset()
    vel_dataset, acc_dataset = dataset.get_imu_statistic()
    detector = BarcodeDetector()

    image_dataset_length = dataset.get_image_sequence().get_length()

    isam = gtsam.ISAM2()

    graph = gtsam.NonlinearFactorGraph()

    imu_params = get_imu_params(vel_dataset[0, :])
    #acc_bias = np.array([0.016, -0.005, -0.045]) # PODUMAT`
    #gyro_bias = np.array([-0.00003, -0.00035, 0.00003])

    acc_bias = np.array([0, 0, 0])  # PODUMAT`
    gyro_bias = np.array([0, 0, 0])

    biases = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias) # very important stuff
    preintegration = gtsam.PreintegratedImuMeasurements(imu_params, biases)

    initial_estimation = gtsam.Values()

    initial_pose = gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0),
                  gtsam.Point3(0, 0, 0))
    initial_speed = np.array([0, 0, 0])
    initial_estimation.insert(B(0), biases)
    initial_estimation.insert(X(0), initial_pose)
    initial_estimation.insert(V(0), initial_speed)

    # add prior on beginning
    prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.01) #dim, roll, pitch, yaw in rad, x, y, z in meters
    vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.01) #dim, sigma in m/s
    bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.00001)

    graph.push_back(gtsam.PriorFactorPose3(X(0), initial_pose, prior_noise)) #from imu_example2
    graph.push_back(gtsam.PriorFactorVector(V(0), initial_speed, vel_noise))
    graph.push_back(gtsam.PriorFactorConstantBias(B(0), biases, bias_noise))

    dt = 1 / 200 # TODO: check or change this according to dataset
    i = 0

    for k in range(vel_dataset.shape[0]):
        measured_gyro_x = vel_dataset[k, 0]
        measured_gyro_y = vel_dataset[k, 1]
        measured_gyro_z = vel_dataset[k, 2]

        measured_gyro_vector = np.array([measured_gyro_x,
                                         measured_gyro_y,
                                         measured_gyro_z])

        measured_acceleration_x = vel_dataset[k, 0]
        measured_acceleration_y = vel_dataset[k, 1]
        measured_acceleration_z = vel_dataset[k, 2]

        measured_acceleration_vector = np.array([measured_acceleration_x,
                                                 measured_acceleration_y,
                                                 measured_acceleration_z])

        preintegration.integrateMeasurement(measured_acceleration_vector, measured_gyro_vector, dt)

        if k % 20 == 0 and k > 0:
        #if k > 0:
            preintegrated_factor = gtsam.ImuFactor(X(i), V(i),
                                     X(i + 1), V(i + 1),
                                     B(0), preintegration)

            graph.push_back(preintegrated_factor)
            graph.push_back(gtsam.BetweenFactorConstantBias(B(i), B(i + 1), biases, bias_noise))

            if k == 20:
                prev_pose = initial_pose
                prev_speed = initial_speed

            #print(k, prev_pose)

            predicted_pose, predicted_speed = get_prediction(prev_pose, prev_speed, biases, preintegration)

            initial_estimation.insert(X(i + 1), predicted_pose) #TODO: find how to predict. see [1]
            initial_estimation.insert(V(i + 1), predicted_speed)
            initial_estimation.insert(B(i + 1), biases)


            '''
            add: graph.push_back(camera_factor)
            add: initial_estimation.insert(camera_points)
            
                // Add node value for feature/landmark if it doesn't already exist
                bool new_landmark = !optimizedNodes.exists(Symbol('l', landmark_id));
                if (new_landmark) {
                  newNodes.insert(landmark, world_point);
                }
                
                // Add factor from this frame's pose to the feature/landmark
                graph.emplace_shared<GenericStereoFactor<Pose3, Point3>>(StereoPoint2(uL, uR, v), pose_landmark_noise, Symbol('x', pose_id), landmark, K);
                    
                // Add prior to the landmark as well    
                graph.emplace_shared<PriorFactor<Point3> >(landmark, world_point, landmark_noise);
                
                double Tx;                    // Camera calibration extrinsic: distance from cam0 to cam1
                gtsam::Matrix4 T_cam_imu_mat; // Transform to get to camera IMU frame from camera frame
            '''

            #isam.update(graph, initial_estimation)
            #result = isam.calculateEstimate()

            optimiser = gtsam.GaussNewtonOptimizer(graph, initial_estimation)
            result = optimiser.optimize()

            prev_pose = result.atPose3(X(i))
            prev_speed = result.atVector(V(i))

            #TODO: initial_pose, initial_speed, biases = [2]

            #plot
            plot.plot_incremental_trajectory(0, result, start=i, scale=1, time_interval=0.0001)

            # reset
            #graph = gtsam.NonlinearFactorGraph()
            #initial_estimation.clear()
            #preintegration.resetIntegration()
            #preintegration.resetIntegrationAndSetBias(prev_robot_bias)
            i += 1

if __name__ == '__main__':
    run()


    """
    [1]
        // Predict initial estimates for current state 
              NavState prev_optimized_state = NavState(prev_robot_pose, prev_robot_velocity);
              NavState propagated_state = imu_preintegrated->predict(prev_optimized_state, prev_robot_bias);
              newNodes.insert(Symbol('x', pose_id), propagated_state.pose()); 
              newNodes.insert(Symbol('v', pose_id), propagated_state.v()); 
              newNodes.insert(Symbol('b', pose_id), prev_robot_bias); 
              
    [2]
        // Get optimized nodes for next iteration 
              prev_robot_pose = optimizedNodes.at<Pose3>(Symbol('x', pose_id));
              prev_robot_velocity = optimizedNodes.at<Vector3>(Symbol('v', pose_id));
              prev_robot_bias = optimizedNodes.at<imuBias::ConstantBias>(Symbol('b', pose_id));
              
    [3]
        // transform landmark coordinates to world frame
              Pose3 prev_camera_pose = Pose3(T_cam_imu_mat) * prev_robot_pose;
              world_point = prev_camera_pose.transform_from(camera_point);
    """

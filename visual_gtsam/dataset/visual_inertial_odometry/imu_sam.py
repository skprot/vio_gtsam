import gtsam
import numpy as np
from visual_gtsam.barcode_detector import BarcodeDetector
from visual_gtsam.dataset import Dataset
from gtsam.symbol_shorthand import B, V, X


def get_imu_params(g=9.81):
    """Create default parameters with Z *up* and realistic noise parameters"""
    params = gtsam.PreintegrationParams.MakeSharedU(g)
    k_gyro_sigma = np.radians(0.5) / 60  # 0.5 degree ARW
    k_accel_sigma = 0.1 / 60  # 10 cm VRW (RANDOM WALK)

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

def run():
    dataset = Dataset()
    vel_dataset, acc_dataset = dataset.get_imu_statistic()
    detector = BarcodeDetector()

    image_dataset_length = dataset.get_image_sequence().get_length()

    isam = gtsam.ISAM2()

    graph = gtsam.NonlinearFactorGraph()

    imu_params = get_imu_params()
    biases = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias) # very important stuff
    preintegration = gtsam.PreintegratedImuMeasurements(imu_params, biases)

    initial_estimation = gtsam.Values()
    initial_estimation.insert(B(0), biases)
    initial_estimation.insert(X(0), initial_pose)
    initial_estimation.insert(V(0), initial_speed)

    # add prior on beginning
    graph.push_back(gtsam.PriorFactorPose3(X(0), initial_pose, prior_noise)) #from imu_example2
    graph.push_back(gtsam.PriorFactorVector(V(0),initial_speed, vel_noise))

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

        if k == 20:
            preintegrated_factor = gtsam.ImuFactor(X(i), V(i),
                                     X(i + 1), V(i + 1),
                                     B(0), preintegration)
            graph.push_back(preintegrated_factor)
            preintegration.resetIntegration()
            initial_estimation.insert(X(i), predicted_pose) #TODO: find how to predict. see [1]
            initial_estimation.insert(V(i), predicted_speed)
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
            isam.update(graph, initial_estimation)
            result = isam.calculateEstimate()

            #TODO: initial_pose, initial_speed, biases = [2]

            # reset
            graph = gtsam.NonlinearFactorGraph()
            initial_estimation.clear()



'''

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


      
'''
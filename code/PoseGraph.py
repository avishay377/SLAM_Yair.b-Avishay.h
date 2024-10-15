import gtsam
from gtsam import symbol
import numpy as np

class PoseGraph:
    def __init__(self, db=None, bundle_data=None, relative_poses=None, cov_matrices=None,
                 key_frames=None):
        self.__db = db
        self.__bundle_data = bundle_data
        self.__relative_poses = relative_poses
        self.__cov_matrices = cov_matrices
        self.__global_poses = []
        self.__key_frames = key_frames
        self.__graph = gtsam.NonlinearFactorGraph()
        self.__initial_estimate = gtsam.Values()
        self.__optimized_values = gtsam.Values()
        self.__camera_sym = []


    def create_factor_graph(self):
        if not self.__db or not self.__bundle_data or not self.__relative_poses or not self.__cov_matrices:
            raise ValueError("db, bundle_data, relative_poses, cov_matrices must be set")
        cur_cam_pose = gtsam.Pose3()
        self.__global_poses.append(cur_cam_pose)
        first_cam_sym = symbol('c', 0)
        self.__camera_sym.append(first_cam_sym)
        sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [1e-1, 1e-2, 1.0])
        pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
        factor = gtsam.PriorFactorPose3(first_cam_sym, cur_cam_pose, pose_uncertainty)
        self.__graph.add(factor)
        self.__initial_estimate.insert(first_cam_sym, cur_cam_pose)
        prev_cam_sym = first_cam_sym
        for i in range(len(self.__relative_poses)):
            #create between factor and add it
            cur_cam_sym = symbol('c', i + 1)
            self.__camera_sym.append(cur_cam_sym)
            pose_uncertainty = gtsam.noiseModel.Gaussian.Covariance(self.__cov_matrices[i])
            factor = gtsam.BetweenFactorPose3(prev_cam_sym, cur_cam_sym, self.__relative_poses[i], pose_uncertainty)
            self.__graph.add(factor)
            #add initial estimate
            cur_cam_pose = cur_cam_pose.compose(self.__relative_poses[i])
            self.__global_poses.append(cur_cam_pose)
            self.__initial_estimate.insert(cur_cam_sym, cur_cam_pose)
            prev_cam_sym = cur_cam_sym

    def optimize(self):
         optimizer = gtsam.LevenbergMarquardtOptimizer(self.__graph, self.__initial_estimate)
         self.__optimized_values = optimizer.optimize()


    def get_optimized_values(self):
        return self.__optimized_values

    def get_initial_estimate(self):
        return self.__initial_estimate


    def marginals(self, optimized=True):
        if optimized:
            return gtsam.Marginals(self.__graph, self.__optimized_values)
        return gtsam.Marginals(self.__graph, self.__initial_estimate)



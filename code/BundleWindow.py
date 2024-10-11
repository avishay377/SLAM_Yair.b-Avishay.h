import gtsam
import numpy as np
from gtsam import symbol
from tracking_database import TrackingDB

from algorithms_library import create_calib_mat_gtsam, create_ext_matrix_gtsam, triangulate_gtsam

LAND_MARK_SYM = "q"
CAMERA_SYM = "c"
P3D_MAX_DIST = 80


class BundleWindow:
    def __init__(self, first_key_frame_id, last_key_frame_id, all_frames_between=True, little_bundle_track=None,
                 db=None):
        self.__first_key_frame_id = first_key_frame_id
        self.__last_key_frame_id = last_key_frame_id
        self.__all_frames_between = all_frames_between

        if all_frames_between:
            self.__bundle_frames = range(first_key_frame_id, last_key_frame_id + 1)
            self.__bundle_len = 1 + last_key_frame_id - first_key_frame_id
        else:
            self.__bundle_frames = [first_key_frame_id, last_key_frame_id]
            self.__bundle_len = 2
            self.__little_bundle_track = little_bundle_track

        self.optimizer = None
        self.__initial_estimate = gtsam.Values()
        self.__optimized_values = None
        self.__camera_sym = set()
        self.__landmark_sym = set()
        self.__tracks = set()
        self.__transformations = []

        self.db = db
        self.graph = gtsam.NonlinearFactorGraph()

    def graph(self):
        return self.graph()

    def get_optimized_values(self):
        return self.__optimized_values
    def create_factor_graph(self):
        print(f"Creating factor graph for frames from {self.__first_key_frame_id} to {self.__last_key_frame_id}")
        self.compose_transformations()

        calib_mat_gtsam = create_calib_mat_gtsam()
        for frame_id in self.__bundle_frames:
            self.__tracks.update(self.db.tracks(frame_id))

        for track in self.__tracks:
            frames_of_track = self.db.frames(track)
            frames_of_track = [frame for frame in frames_of_track if
                               self.__first_key_frame_id <= frame <= self.__last_key_frame_id]
            link = self.db.link(frameId=frames_of_track[-1], trackId=track)
            p3d_gtsam = triangulate_gtsam(self.__transformations[frames_of_track[-1] - self.__first_key_frame_id],
                                          calib_mat_gtsam, link)
            landmark_gtsam_symbol = symbol(LAND_MARK_SYM, track)
            self.__initial_estimate.insert(landmark_gtsam_symbol, p3d_gtsam)
            self.__landmark_sym.add(landmark_gtsam_symbol)

            for frame_id in frames_of_track:
                # Factor creation

                frame_l_xy = self.db.link(frame_id, track).left_keypoint()
                frame_r_xy = self.db.link(frame_id, track).right_keypoint()

                gtsam_measurement_pt2 = gtsam.StereoPoint2(frame_l_xy[0], frame_r_xy[0], frame_l_xy[1])
                projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
                camera_symbol_gtsam = symbol("c", frame_id - self.__first_key_frame_id)
                factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                                     camera_symbol_gtsam, symbol(LAND_MARK_SYM, track),
                                                     calib_mat_gtsam)
                self.__camera_sym.add(camera_symbol_gtsam)
                self.graph.add(factor)

    def compose_transformations(self):

        for i in range(len(self.__bundle_frames)):

            if i == 0:
                current_Rt_gtsam = gtsam.Pose3()
                prev = current_Rt_gtsam

                self.__transformations.append(current_Rt_gtsam)

                camera_symbol_gtsam = symbol(CAMERA_SYM, i)
                self.__initial_estimate.insert(camera_symbol_gtsam, current_Rt_gtsam)
                self.__camera_sym.add(camera_symbol_gtsam)
                sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [1e-1, 1e-2, 1.0])
                pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)  # todo: what about frame[0](0,0,0)

                # Initial pose
                camera_symbol_gtsam = symbol(CAMERA_SYM, 0)
                factor = gtsam.PriorFactorPose3(camera_symbol_gtsam, current_Rt_gtsam, pose_uncertainty)
                self.__camera_sym.add(camera_symbol_gtsam)
                self.graph.add(factor)

            else:
                #todo: avish tried compose differently
                # convert to inverse by gtsam.Pose3.inverse
                current_left_transformation_homogenus = self.compose_to_first_kf(self.__bundle_frames[i])
                Rt_inverse_gtsam = (gtsam.Pose3.inverse(gtsam.Pose3(current_left_transformation_homogenus)))
                self.__transformations.append(Rt_inverse_gtsam)
                camera_symbol_gtsam = symbol(CAMERA_SYM, i)
                self.__initial_estimate.insert(camera_symbol_gtsam, Rt_inverse_gtsam)

                self.__camera_sym.add(camera_symbol_gtsam)



                #
                # compose_curr_Rt_gtsam = current_Rt_gtsam.compose(prev)
                # prev = compose_curr_Rt_gtsam
                # self.__transformations.append(compose_curr_Rt_gtsam)
                # camera_symbol_gtsam = symbol(CAMERA_SYM, i)
                # self.__initial_estimate.insert(camera_symbol_gtsam, compose_curr_Rt_gtsam)
                # self.__camera_sym.add(camera_symbol_gtsam)
        return



    def calculate_graph_error(self, optimized=True):

        if not optimized:
            error = self.graph.error(self.__initial_estimate)
        else:
            error = self.graph.error(self.__optimized_values)

        return np.log(error)

    def optimize(self):
        """
        Apply optimization with Levenberg marquardt algorithm
        """
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.__initial_estimate)
        self.__optimized_values = self.optimizer.optimize()

    def get_initial_estimate(self):
        return self.__initial_estimate

    def get_optimized_values(self):
        return self.__optimized_values

    def get_cameras_symbols_lst(self):
        """
        Return cameras symbols list
        """
        return self.__camera_sym

    def get_landmarks_symbols_lst(self):
        """
        Returns landmarks symbols list
        """
        return self.__landmark_sym


    def get_optimized_cameras(self):

        cams = []
        for frame_id in range(self.__first_key_frame_id, self.__last_key_frame_id + 1):
            cams.append(self.__optimized_values.atPose3(symbol(CAMERA_SYM, frame_id)))
        return cams


    def get_optimized_last_camera(self):
        """
        Get the optimized last camera
        Returns:
            Pose3: The optimized last camera
        """
        return self.__optimized_values.atPose3(symbol(CAMERA_SYM, self.__last_key_frame_id - self.__first_key_frame_id))

    def get_optimized_landmarks(self):
        landmarks = []
        for landmark_sym in self.__landmark_sym:
            landmarks.append(self.__optimized_values.atPoint3(landmark_sym))

        return np.asarray(landmarks)


    def get_optimized_landmarks_lst(self):
        """
        Get the optimized landmarks
        Returns:
            list: The optimized landmarks
        """
        landmarks = []
        for landmark_sym in self.__landmark_sym:
            landmark = self.__optimized_values.atPoint3(landmark_sym)
            landmarks.append(landmark)
        return landmarks

    #todo: function that avish added
    def get_homogeneous_transformation(self, frame_id):
        matrix = self.db.rotation_matrices[frame_id]
        translation = self.db.translation_vectors[self.__first_key_frame_id]
        translation = translation.reshape(3, 1)
        R = np.concatenate((matrix, np.zeros((1, 3))), axis=0)
        zero_row = np.array([0]).reshape(1, 1)
        translation_homogenus = np.vstack((translation, zero_row))
        # transformation_homogenus = np.concatenate((R, translation_homogenus.reshape(4, 1)), axis=1)
        return R, translation_homogenus

    def get_frame_range(self):
        return (self.__first_key_frame_id, self.__last_key_frame_id)

    def compose_to_first_kf(self, frame_id):
        kf_R, kf_t = self.get_homogeneous_transformation(self.__first_key_frame_id)
        frame_R, frame_t = self.get_homogeneous_transformation(frame_id)
        compose_R = kf_R.T @ frame_R
        compose_t = kf_R.T @ (frame_t - kf_t)
        compose_Rt = np.concatenate((compose_R, compose_t), axis=1)
        return compose_Rt

    def get_landmarks_symbols_set(self):
        """
        Returns landmarks symbols list
        """
        return self.__landmark_sym

def solve_bundle_window(db, first_frame, last_frame):
    window = BundleWindow(first_frame, last_frame, db=db)
    window.create_factor_graph()
    error_before_optim = window.calculate_graph_error(False)
    window.optimize()
    error_after_optim = window.calculate_graph_error()
    return window, error_before_optim, error_after_optim

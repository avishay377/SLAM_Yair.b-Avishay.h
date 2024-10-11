import numpy as np
import gtsam
from gtsam import symbol
from tracking_database import TrackingDB
from BundleWindow import solve_bundle_window



def convert_landmarks_cam_rel_to_global(camera, landmarks):
    glob_landmarks = []
    for landmark in landmarks:
        glob_landmark = camera.transformFrom(landmark)
        glob_landmarks.append(glob_landmark)
    return glob_landmarks

def convert_landmarks_to_global(cams, landmarks):
    glob_landmarks = []
    for camera, cam_landmarks in zip(cams, landmarks):
        glob_landmarks_camera = convert_landmarks_cam_rel_to_global(camera, cam_landmarks)
        glob_landmarks += glob_landmarks_camera

    return np.array(glob_landmarks)


def convert_gtsam_cams_to_global(arr):
    relative_arr = []
    last = arr[0]

    for t in arr:
        last = last.compose(t)
        relative_arr.append(last)

    return relative_arr

class BundleAdjustment:

    __bundle_windows = None
    __first_key_frame_id = None
    __last_key_frame_id = None

    db = None


    def __init__(self, db, first_frame_id, last_frame_id):
        self.__bundle_windows = []
        self.__first_key_frame_id = first_frame_id
        self.__last_key_frame_id = last_frame_id
        self.__gtsam_cams_bundle = None
        self.__gtsam_landmarks_bundle = None
        self.__key_frames = None
        self.db = db

    def solve_with_window_size_20(self):
        self.__key_frames = [0]
        for i in range(self.__first_key_frame_id, self.__last_key_frame_id, 20):
            if (i < self.__last_key_frame_id - 20):
                curr_window, error_curr_window_before_opt, error_curr_window_after_opt = solve_bundle_window(self.db, i, i + 20)
                self.__bundle_windows.append(curr_window)
                self.__key_frames.append(i + 20)
            else:
                curr_window, error_curr_window_before_opt, error_curr_window_after_opt = solve_bundle_window(self.db, i, self.__last_key_frame_id)
                self.__bundle_windows.append(curr_window)
                self.__key_frames.append(self.__last_key_frame_id)
        try:
            print(f"The error of the last window is:{error_curr_window_after_opt}")
        except:
            print("No window was solved?")
        try:
            values_last_window = curr_window.get_optimized_values()
        except:
            print("No window was solved?")
            return
        kf_key = values_last_window.keys()[0]
        kf = values_last_window.atPose3(kf_key)
        print(f"Keyframe matrix: {kf}")

        cams = [gtsam.Pose3()]
        self.__gtsam_landmarks_bundle = []\

        for window in self.__bundle_windows:
            cams.append(window.get_optimized_last_camera())
            self.__gtsam_landmarks_bundle.append(window.get_optimized_landmarks_lst())
        self.__gtsam_cams_bundle = np.array(cams)




    def get_key_frames(self):
        return self.__key_frames


    def get_gtsam_cams(self):
        return self.__gtsam_cams_bundle

    def get_gtsam_landmarks(self):
        return self.__gtsam_landmarks_bundle


    def get_last_window(self):
        return self.__bundle_windows[-1]

    def get_all_optimized_values(self):
        #organize it to take the values for each window and save it as a list of lists of values

        all_values = []
        for window in self.__bundle_windows:
            window_values = window.get_optimized_values()
            all_values.append(window_values)
        return all_values

        #
        # all_values = gtsam.Values()
        # for window in self.__bundle_windows:
        #     window_values = window.get_optimized_values()
        #     for key in window_values.keys():
        #         if not all_values.exists(key):
        #             if gtsam.Symbol(key).chr() == 'c':
        #                 all_values.insert(key, window_values.atPose3(key))
        #             elif gtsam.Symbol(key).chr() == 'q':
        #                 all_values.insert(key, window_values.atPoint3(key))
        # return all_values

    def get_windows(self):
        return self.__bundle_windows
    # def operator[](self, i):
    #     return self.__bundle_windows[i]

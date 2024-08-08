import numpy as np
import gtsam
from gtsam import symbol
from tracking_database import TrackingDB
from BundleWindow import solve_bundle_window


class BundleAdjustment:

    __bundle_windows = None
    __first_key_frame_id = None
    __last_key_frame_id = None

    db = None


    def __init__(self, db, first_frame_id, last_frame_id):
        self.__bundle_windows = []
        self.__first_key_frame_id = first_frame_id
        self.__last_key_frame_id = last_frame_id
        self.db = db

    def solve_with_window_size_20(self):
        for i in range(self.__first_key_frame_id, self.__last_key_frame_id, 20):
            if (i < self.__last_key_frame_id - 20):
                curr_window, error_curr_window_before_opt, error_curr_window_after_opt = solve_bundle_window(self.db, i, i + 20)
            else:
                curr_window, error_curr_window_before_opt, error_curr_window_after_opt = solve_bundle_window(self.db, i, self.__last_key_frame_id)
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
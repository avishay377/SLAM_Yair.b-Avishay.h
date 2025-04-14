import gtsam
from BundleAdjustment import BundleAdjustment
from PoseGraph import PoseGraph
from ex6 import get_relative_pose_and_cov_mat_last_kf
from tracking_database import TrackingDB
from BundleWindow import BundleWindow
from gtsam.utils import plot
from gtsam import symbol
import numpy as np
import matplotlib.pyplot as plt
from VertexGraph import VertexGraph
import pickle
from algorithms_library import (read_images_from_dataset ,extract_keypoints_and_inliers,
                                calculate_right_camera_matrix, cv_triangulate_matched_points,
                                find_consensus_matches_indices, )
import cv2


CANDIDATES_THRESHOLD = 5
DETECTOR = cv2.SIFT_create()


def ransac_candidates(kf, candidates):
    # manipulation to the index
    img_kf = read_images_from_dataset(kf)
    keypoints_kf, desc_kf = DETECTOR.detectAndCompute(img_kf)
    for candidate in candidates:

        ind = candidate
        img_candidate =



def estimate_cov_matrix(path, cov_matrices):
    cov_mat = cov_matrices[path[0]]
    for pose in range(1, len(path)):
        cov_mat = cov_mat + cov_matrices[path[pose]]
    return cov_mat
def q1(db, poseGraph_saved=False):

    symbol_c = "c"
    # use ex6.py methods to get the covariance between cosecutive cameras
    bundle = BundleAdjustment(db)
    bundle.load("bundle_data_window_size_20_witohut_bad_matches_ver2/",
                "bundle with window_size_20_witohut_bad_matches_ver2", 5)

    relative_poses, cov_matrices = [], []
    for i, window in enumerate(bundle.get_windows()):
        if i % 100 == 0:
            print(f"try to get relative pose and cov mat for window {i}")
        relative_pose, cov_matrix = get_relative_pose_and_cov_mat_last_kf(window)
        relative_poses.append(relative_pose)
        cov_matrices.append(cov_matrix)
    poseGraph = PoseGraph(db, bundle, relative_poses, cov_matrices, bundle.get_key_frames())
    poseGraph.create_factor_graph()
    # if not poseGraph_saved:
    #     # save the poseGraph using pickle
    #     poseGraph.optimizer = None
    #     filehandler = open("poseGraph.pkl", "wb")
    #     pickle.dump(poseGraph, filehandler)
    #     filehandler.close()
    # else:
    #     # load the poseGraph using pickle
    #     filehandler = open("poseGraph.pkl", "rb")
    #     poseGraph = pickle.load(filehandler)
    #     filehandler.close()
    values = poseGraph.get_initial_estimate()
    vertexGraph = VertexGraph(len(cov_matrices), cov_matrices)
    for n in range(len(cov_matrices)):
        candidates = find_candidates(cov_matrices, n, relative_poses, symbol_c, values, vertexGraph)
        # iterate over candidates and check if we got consective-matches by ransac and SIFT etc..
        # todo: check wether the index of keframe should match to the index in the dataset of the images




def find_candidates(cov_matrices, n, relative_poses, symbol_c, values, vertexGraph):
    symbol_n = symbol_c + f"{n}"
    # todo: maybe?
    # symbol_n = symbol(symbol_c, n)
    candidates = []
    for i in range(n):
        dist, path = vertexGraph.find_shortest_path(i, n)
        cov_mat_in = estimate_cov_matrix(path, cov_matrices)
        symbol_i = symbol_c + f"{i}"
        # todo: maybe?
        # symbol_i = symbol(symbol_c, i)
        identity_mat = np.identity(3)
        # add col of 0
        identity_mat = np.column_stack((identity_mat, np.zeros((3, 1))))
        delta_x = gtsam.BetweenFactorPose3(symbol_i, symbol_n, identity_mat, cov_mat_in)
        error = delta_x.error(values)
        if error < CANDIDATES_THRESHOLD:
            candidates.append((i, error))
    # sort candidates by error
    candidates.sort(key=lambda x: x[1])
    # take the first 3 candidates
    candidates = candidates[:3]
    return candidates



if __name__ == '__main__':
    db = TrackingDB()
    db.load('db')
    q1(db)

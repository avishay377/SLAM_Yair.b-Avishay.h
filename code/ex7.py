import gtsam
import math
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
from algorithms_library import (read_images_from_dataset, extract_keypoints_and_inliers,
                                calculate_right_camera_matrix, cv_triangulate_matched_points,
                                find_consensus_matches_indices, read_cameras_matrices, )
import cv2
from ex3 import q1 as ex3_q1
from ex3 import q2 as ex3_q2
from ex3 import q3 as ex3_q3
from ex3 import q4 as ex3_q4
from ex3 import q5 as ex3_q5



CANDIDATES_THRESHOLD = 5
TRESHOLD_INLIERS = 80
DETECTOR = cv2.SIFT_create()
CAMERA_SYM = "c"

def q2(kf, candidates):
    # manipulation to the index
    valid_candidates = []
    img0_left, img0_right = read_images_from_dataset(kf * 5)
    for cand in candidates:
        img1_left, img1_right = read_images_from_dataset(cand * 5)
        K, Ext0_left, Ext0_right = read_cameras_matrices()  # intrinsic & extrinsic camera Matrices
        R0_left, t0_left = Ext0_left[:, :3], Ext0_left[:, 3:]
        R0_right, t0_right = Ext0_right[:, :3], Ext0_right[:, 3:]

        # QUESTION 1
        q1_output = ex3_q1(K, R0_left, R0_right, img0_left, img0_right, img1_left, img1_right, t0_left, t0_right)
        (descriptors0_left, descriptors1_left, inliers_0_0, inliers_1_1, keypoints0_left, keypoints0_right, keypoints1_left,
         keypoints1_right, point_cloud_0) = q1_output
        # QUESTION 2
        tracking_matches = ex3_q2(descriptors0_left, descriptors1_left)
        # QUESTION 3
        q3_output = ex3_q3(Ext0_left, Ext0_right, K, R0_right, inliers_0_0, inliers_1_1, keypoints1_left, point_cloud_0,
                       t0_right, tracking_matches)
        R1_left, R1_right, consensus_match_indices_0_1, t1_left, t1_right = q3_output
        # QUESTION 4
        ex3_q4(K, R0_left, R0_right, R1_left, R1_right, consensus_match_indices_0_1, img0_left, img1_left, inliers_0_0,
           inliers_1_1, keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0, t0_left,
           t0_right, t1_left, t1_right, tracking_matches)
        # QUESTION 5:
        inliers_perc = ex3_q5(K, R0_left, R0_right, consensus_match_indices_0_1, img0_left, img1_left, inliers_0_0, inliers_1_1,
           keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0, t0_left, t0_right,
           tracking_matches)
        if inliers_perc > TRESHOLD_INLIERS:
            valid_candidates.append(cand)
    return valid_candidates


def mahalanobis_dist(delta, cov):
    r_squared = delta.T @ np.linalg.inv(cov) @ delta
    return r_squared ** 0.5


def estimate_cov_matrix(path, cov_matrices):
    cov_mat = cov_matrices[path[0]]
    for pose in range(1, len(path)):
        cov_mat = cov_mat + cov_matrices[path[pose]]
    return cov_mat



def main(db, poseGraph_saved=False):
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

    for n in range(len(cov_matrices)):
        candidates = q1(n, poseGraph, cov_matrices, relative_poses, symbol_c)
        if len(candidates) > 0:
            candidates_ind = [candidates[i][0] for i in range(len(candidates))]
            valid_candidates = q2(n, candidates_ind)
            print(f"Window number {n}: {valid_candidates}")
        else:
            print(f"Window number {n}: No valid candidates")


def q1(n, poseGraph, cov_matrices, relative_poses, symbol_c):
    values = poseGraph.get_initial_estimate()
    vertexGraph = VertexGraph(len(cov_matrices), cov_matrices)
    candidates = find_candidates(cov_matrices, n, relative_poses, symbol_c, values, vertexGraph, poseGraph)
        # iterate over candidates and check if we got consective-matches by ransac and SIFT etc..
        # todo: check wether the index of keframe should match to the index in the dataset of the images
    return candidates




# def find_candidates(cov_matrices, n, relative_poses, symbol_c, values, vertexGraph):
#     candidates = []
#     symbol_n = symbol(symbol_c, n)
#
#     for i in range(n):
#         dist, path = vertexGraph.find_shortest_path(i, n)
#         cov_mat_in = estimate_cov_matrix(path, cov_matrices)
#         # symbol_i = symbol_c + f"{i}"
#         symbol_i = symbol(symbol_c, i)
#         # todo: maybe?
#         # symbol_i = symbol(symbol_c, i)
#         # identity_mat = np.identity(3)
#         # add col of 0
#         # identity_mat = np.column_stack((identity_mat, np.zeros((3, 1))))
#         delta_x = gtsam.BetweenFactorPose3(symbol_i, symbol_n, relative_poses[i], cov_matrices[i])
#         error = delta_x.error(values)
#         if error < CANDIDATES_THRESHOLD:
#             candidates.append((i, error))
#     # sort candidates by error
#     candidates.sort(key=lambda x: x[1])
#     # take the first 3 candidates
#     candidates = candidates[:3]
#     return candidates

def gtsam_cams_delta(first_cam_mat, second_cam_mat):
    gtsam_rel_trans = second_cam_mat.between(first_cam_mat)
    return gtsam_translation_to_vec(gtsam_rel_trans.rotation(), gtsam_rel_trans.translation())


def gtsam_translation_to_vec(R_mat, t_vec):
    np_R_mat = np.hstack((R_mat.column(1).reshape(3, 1), R_mat.column(2).reshape(3, 1), R_mat.column(3).reshape(3, 1)))
    euler_angles = rot_mat_to_euler_angles(np_R_mat)
    return np.hstack((euler_angles, t_vec))


def rot_mat_to_euler_angles(R_mat):
    sy = math.sqrt(R_mat[0, 0] * R_mat[0, 0] + R_mat[1, 0] * R_mat[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R_mat[2, 1], R_mat[2, 2])
        y = math.atan2(-R_mat[2, 0], sy)
        z = math.atan2(R_mat[1, 0], R_mat[0, 0])
    else:
        x = math.atan2(-R_mat[1, 2], R_mat[1, 1])
        y = math.atan2(-R_mat[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def find_candidates(cov_matrices, n, relative_poses, symbol_c, values, vertexGraph, poseGraph):
    candidates = []
    cur_cam_mat = values.atPose3(symbol(CAMERA_SYM, n))  # 'n' camera : cur_cam -> world

    for prev_cam_pose_graph_ind in range(n):  # Run on the previous cameras 0 <= i < n

        prev_cam_mat = values.atPose3(symbol(CAMERA_SYM, prev_cam_pose_graph_ind))  # 'i' camera : prev_cam -> world

        # Find the shortest path and estimate its relative covariance
        shortest_path = vertexGraph.find_shortest_path(prev_cam_pose_graph_ind, n)
        estimated_rel_cov = vertexGraph.estimate_rel_cov(shortest_path)

        # Compute Cams delta and their mahalanobis distance
        cams_delta = gtsam_cams_delta(prev_cam_mat, cur_cam_mat)
        dist = mahalanobis_dist(cams_delta, estimated_rel_cov)

        if dist < 50:
            candidates.append([dist, prev_cam_pose_graph_ind])

    # if there are candidates, choose the best MAX_CAND_NUM numbers
    if len(candidates) > 0:
        # print(cur_cam_pose_graph_ind, candidates)

        sorted_candidates = sorted(candidates, key=lambda elem: elem[0])  # Sort candidates by mahalanobis dist
        # Take only the MAX_CAND_NUM candidate number and the index from the original list (without dist)
        candidates = np.array(sorted_candidates[:3]).astype(int)[:, 1]

    return candidates



if __name__ == '__main__':
    db = TrackingDB()
    db.load('db')
    main(db)

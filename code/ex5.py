import random
# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import gtsam
import cv2
import numpy as np
from gtsam import symbol

from algorithms_library import (compute_trajectory_and_distance_avish_test, plot_root_ground_truth_and_estimate,
                                read_images_from_dataset, read_cameras_matrices, read_poses, read_poses_truth,
                                xy_triangulation, project, get_euclidean_distance, plot_tracks, calculate_statistics,
                                print_statistics, init_db, compose_transformations, project_point,
                                read_cameras_matrices_linux)
import matplotlib.pyplot as plt
from tracking_database import TrackingDB
import pybind11

NUM_FRAMES = 3360  # or any number of frames you want to process


def projection_factors_error(factors, values):
    errors = []
    for factor in factors:
        errors.append(factor.error(values))

    return np.array(errors)


def add_projection_to_factor(factor, StereoPt, sigma, camera, PtLink, K):
    # if not factor:
    factor = gtsam.GenericStereoFactor3D(StereoPt, sigma, camera, PtLink, K)
    # todo: add the right omg factor of the frame
    # else:
    #     factor.ad


def q1_avish(db: TrackingDB):
    factors = []
    values = gtsam.Values()

    valid_tracks = [track for track in db.all_tracks() if len(db.frames(track)) >= 10]
    selected_track = random.choice(valid_tracks)
    selected_track_frames = db.frames(selected_track)
    frame_ids = [frame_id for frame_id in db.frames(selected_track)]
    track_length = len(frame_ids)

    # Triangulate the 3D point using the last frame of the track
    link = db.link(selected_track_frames[-1], selected_track)
    # Read transformations
    truth_transformations = read_poses_truth(seq=(frame_ids[0], frame_ids[-1] + 1))
    last_left_img_xy = link.left_keypoint()
    last_right_img_xy = link.right_keypoint()
    last_left_truth_transformation = truth_transformations[-1]
    current_left_rotation = db.rotation_matrices[frame_ids[-1]]
    current_left_translation = db.translation_vectors[frame_ids[-1]]

    K_, P_left, P_right = read_cameras_matrices()
    # calculate gtsam.K as the David said.
    # todo: check the skew, and base line parameters, now initial at 0.2
    baseline = P_right[0, 3]
    K = gtsam.Cal3_S2Stereo(K_[0, 0], K_[1, 1], K_[0, 1], K_[0, 2], K_[1, 2], -baseline)
    # todo: check if inverse get R|last_left_translation_homogenus or R and last_left_translation_homogenus seperately. now it is R|last_left_translation_homogenus
    # concat row of zeros to R to get 4x3 matrix
    R = np.concatenate((current_left_rotation, np.zeros((1, 3))), axis=0)

    current_left_translation = current_left_translation.reshape(3, 1)
    # concat last_left_translation_homogenus with zero to get 4x1 vector
    zero_row = np.array([0]).reshape(1, 1)
    current_left_translation_homogenus = np.vstack((current_left_translation, zero_row))
    # last_left_translation_homogenus = np.concatenate((last_left_translation, np.array([0])), axis=0)
    # concat R and last_left_translation_homogenus to get 4x4 matrix
    current_left_transformation_homogenus = np.concatenate((R, current_left_translation_homogenus.reshape(4, 1)),
                                                           axis=1)
    current_left_transformation = np.hstack((current_left_rotation, current_left_translation))
    # P_right_homogens = np.vstack((P_right, np.array([0, 0, 0, 1])))
    current_right_transformation = compose_transformations(current_left_transformation, P_right)
    # convert to inverse by gtsam.Pose3.inverse
    Rt_inverse_gtsam = (gtsam.Pose3.inverse(gtsam.Pose3(current_left_transformation_homogenus)))
    # last_left_translation_homogenus = (gtsam.Pose3.inverse(last_left_transformation_homogenus
    Rt_right_inverse_gtsam = (gtsam.Pose3.inverse(gtsam.Pose3(current_right_transformation)))
    current_frame_camera_left = gtsam.StereoCamera(Rt_inverse_gtsam, K)
    # maybe
    # current_frame_camera_right = gtsam.StereoCamera(Rt_right_inverse_gtsam, K)
    # meassurement = gtsam.Point2Vector([last_left_img_xy, last_right_img_xy])
    stereo_pt_gtsam = gtsam.StereoPoint2(last_left_img_xy[0], last_right_img_xy[0], last_left_img_xy[1])
    triangulate_p3d_gtsam = current_frame_camera_left.backproject(stereo_pt_gtsam)

    # Update values dictionary
    p3d_sym = symbol("q", 0)
    values.insert(p3d_sym, triangulate_p3d_gtsam)

    # calculate re-projection
    left_projections = []
    right_projections = []
    errors_projection_left = []
    errors_projection_right = []
    frames_cameras_gtsam = []
    selected_track_frames = selected_track_frames[::-1]
    frame_ids = frame_ids[::-1]

    frames_l_xy = [db.link(frame, selected_track).left_keypoint() for frame in selected_track_frames]
    frames_r_xy = [db.link(frame, selected_track).right_keypoint() for frame in selected_track_frames]

    # for q2
    graph = gtsam.NonlinearFactorGraph()
    sigma = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))
    cameras = []
    project_points = []
    factors = []
    for i in range(track_length):
        link = db.link(selected_track_frames[i], selected_track)

        # Read transformations
        last_left_img_xy = link.left_keypoint()
        last_right_img_xy = link.right_keypoint()
        last_left_truth_transformation = truth_transformations[-1]
        current_left_rotation = db.rotation_matrices[frame_ids[i]]
        current_left_translation = db.translation_vectors[frame_ids[i]]

        # calculate gtsam.K as the David said.
        # todo: check the skew, and base line parameters, now initial at 0.2
        # todo: check if inverse get R|last_left_translation_homogenus or R and last_left_translation_homogenus seperately. now it is R|last_left_translation_homogenus
        # concat row of zeros to R to get 4x3 matrix
        R = np.concatenate((current_left_rotation, np.zeros((1, 3))), axis=0)

        current_left_translation = current_left_translation.reshape(3, 1)
        # concat last_left_translation_homogenus with zero to get 4x1 vector
        zero_row = np.array([0]).reshape(1, 1)
        current_left_translation_homogenus = np.vstack((current_left_translation, zero_row))

        # last_left_translation_homogenus = np.concatenate((last_left_translation, np.array([0])), axis=0)
        # concat R and last_left_translation_homogenus to get 4x4 matrix
        current_left_transformation_homogenus = np.concatenate((R, current_left_translation_homogenus.reshape(4, 1)),
                                                               axis=1)

        current_left_transformation = np.hstack((current_left_rotation, current_left_translation))
        # P_right_homogens = np.vstack((P_right, np.array([0, 0, 0, 1])))
        current_right_transformation = compose_transformations(current_left_transformation, P_right)
        # convert to inverse by gtsam.Pose3.inverse
        Rt_inverse_gtsam_current = (gtsam.Pose3.inverse(gtsam.Pose3(current_left_transformation_homogenus)))
        # last_left_translation_homogenus = (gtsam.Pose3.inverse(last_left_transformation_homogenus
        Rt_right_inverse_gtsam_current = (gtsam.Pose3.inverse(gtsam.Pose3(current_right_transformation)))
        current_frame_camera_left = gtsam.StereoCamera(Rt_inverse_gtsam_current, K)
        current_frame_camera_right = gtsam.StereoCamera(Rt_right_inverse_gtsam_current, K)
        project_stereo_pt_gtsam = current_frame_camera_left.project(triangulate_p3d_gtsam)
        # project_right = current_frame_camera_right.project(triangulate_p3d_gtsam)

        left_projections.append([project_stereo_pt_gtsam.uL(), project_stereo_pt_gtsam.v()])
        right_projections.append([project_stereo_pt_gtsam.uR(), project_stereo_pt_gtsam.v()])

        left_pose_sym = symbol("c", frame_ids[i])
        values.insert(left_pose_sym, Rt_inverse_gtsam_current)

        # Factor creation
        gtsam_measurement_pt2 = gtsam.StereoPoint2(frames_l_xy[i][0], frames_r_xy[i][0], frames_l_xy[i][1])
        projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                             symbol("c", frame_ids[i]), p3d_sym, K)
        factors.append(factor)

        # qs_i = (gtsam.symbol('q', 2 * (i - 1)), gtsam.symbol('q', 2 * i))
        # todo shoud return the factor for left and add the right factor
        # add_projection_to_factor(project_stereo_pt_gtsam, sigma, cameras_i[0], qs_i[0], K)
        # add_projection_to_factor(factor, project_stereo_pt_gtsam, sigma, cameras_i[1], qs_i[1], K)
        # cameras.append(cameras_i)
        # project_points.append(qs_i)

    left_proj_dist = get_euclidean_distance(np.array(left_projections), np.array(frames_l_xy))
    right_proj_dist = get_euclidean_distance(np.array(right_projections), np.array(frames_r_xy))
    total_proj_dist = (left_proj_dist + right_proj_dist) / 2

    # Factor error
    factor_projection_errors = projection_factors_error(factors, values)

    # Plots re-projection error
    # utils.plot.plot_re_projection_error_graph(factor_projection_errors, frame_idx_triangulate, "")
    plot_factor_re_projection_error_graph(factor_projection_errors, total_proj_dist, -1)
    plot_factor_as_func_of_re_projection_error_graph(factor_projection_errors, total_proj_dist, -1)
    plot_reprojection_error(left_proj_dist, right_proj_dist, selected_track)

    # q2:
    # initialEstimate = gtsam.Values()
    # initialEstimate.insert(cameras[0][0], Rt_inverse_gtsam)
    # initialEstimate.insert(cameras[])


def plot_reprojection_error(left_proj_dist, right_proj_dist, selected_track):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f"Reprojection error for track: {selected_track}")
    # Plotting the scatter plot
    ax.scatter(range(len(left_proj_dist)), left_proj_dist, color='orange', label='Left Projections')
    ax.scatter(range(len(right_proj_dist)), right_proj_dist, color='blue', label='Right Projections')
    # Plotting the continuous line
    ax.plot(range(len(left_proj_dist)), left_proj_dist, linestyle='-', color='orange')
    ax.plot(range(len(right_proj_dist)), right_proj_dist, linestyle='-', color='blue')
    ax.set_ylabel('Error')
    ax.set_xlabel('Frames')
    ax.legend()
    fig.savefig("Reprojection_error.png")
    plt.close(fig)


def plot_factor_re_projection_error_graph(factor_projection_errors, total_proj_dist, frame_idx_triangulate):
    """
    Plots re projection error
    """
    exponent_vals = np.exp(0.5 * total_proj_dist)
    fig, ax = plt.subplots(figsize=(10, 7))

    frame_title = "Last" if frame_idx_triangulate == -1 else "First"

    ax.set_title(f"Factor Re projection error from {frame_title} frame")
    plt.scatter(range(len(factor_projection_errors)), factor_projection_errors, label="Factor")
    # plt.scatter(range(len(exponent_vals)), exponent_vals, label="exp(0.5* ||z - proj(c, q)||)")
    plt.legend(loc="upper right")
    plt.ylabel('Error')
    plt.xlabel('Frames')

    fig.savefig(f"Factor Re projection error graph for {frame_title} frame.png")
    plt.close(fig)


def plot_factor_as_func_of_re_projection_error_graph(factor_projection_errors, total_proj_dist,  frame_idx_triangulate):
    """
    Plots re projection error
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    frame_title = "Last" if frame_idx_triangulate == -1 else "First"

    ax.set_title(f"Factor error as a function of a Re projection error graph for {frame_title} frame")
    plt.plot(total_proj_dist, factor_projection_errors, label="Factor")
    plt.plot(total_proj_dist, 0.5 * total_proj_dist ** 2, label="0.5x^2")
    plt.plot(total_proj_dist, total_proj_dist ** 2, label="x^2")
    plt.legend(loc="upper left")
    plt.ylabel('Factor error')
    plt.xlabel('Re projection error')

    fig.savefig(f"Factor error as a function of a Re projection error graph for {frame_title} frame.png")
    plt.close(fig)


def q3():
    pass


if _name_ == '_main_':
    db = TrackingDB()
    db.load('db')
    q1_avish(db)
    q3()
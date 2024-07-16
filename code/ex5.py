import random
import matplotlib.pyplot as plt
import numpy as np
import gtsam
import cv2
import numpy as np
from algorithms_library import (compute_trajectory_and_distance_avish_test, plot_root_ground_truth_and_estimate,
                                read_images_from_dataset, read_cameras_matrices, read_poses, read_poses_truth,
                                xy_triangulation, project, get_euclidean_distance, plot_tracks, calculate_statistics,
                                print_statistics, init_db, compose_transformations, project_point,
                                read_cameras_matrices_linux)
import matplotlib.pyplot as plt
from tracking_database import TrackingDB
import pybind11


NUM_FRAMES = 3360  # or any number of frames you want to process


def q1_avish(db: TrackingDB):
    valid_tracks = [track for track in db.all_tracks() if len(db.frames(track)) >= 10]
    # Select a random track of length >= 10
    selected_track = random.choice(valid_tracks)
    selected_track_frames = db.frames(selected_track)
    frame_ids = [frame_id for frame_id in db.frames(selected_track)]
    track_length = len(frame_ids)

    # Triangulate the 3D point using the last frame of the track
    link = db.link(selected_track_frames[-1], selected_track)
    # Read transformations
    truth_transformations = read_poses_truth(seq=(frame_ids[0], frame_ids[-1] + 1))

    rotations = db.rotation_matrices
    translations = db.translation_vectors
    last_left_img_xy = link.left_keypoint()
    last_right_img_xy = link.right_keypoint()
    last_left_truth_transformation = truth_transformations[-1]
    last_left_rotation = rotations[frame_ids[-1]]
    last_left_translation = translations[frame_ids[-1]]

    K_, P_left, P_right = read_cameras_matrices()
    # calculate gtsam.K as the David said.
    #todo: check the skew, and base line parameters, now initial at 0.2
    K = gtsam.Cal3_S2Stereo(K_[0, 0], K_[1, 1], 0.2 ,K_[0, 2], K_[1, 2], 0.2)
    # todo: check if inverse get R|last_left_translation_homogenus or R and last_left_translation_homogenus seperately. now it is R|last_left_translation_homogenus
    # concat row of zeros to R to get 4x3 matrix
    R = np.concatenate((last_left_rotation, np.zeros((1, 3))), axis=0)

    last_left_translation = last_left_translation.reshape(3,1)
    # concat last_left_translation_homogenus with zero to get 4x1 vector
    zero_row = np.array([0]).reshape(1, 1)
    last_left_translation_homogenus = np.vstack((last_left_translation, zero_row))

    # last_left_translation_homogenus = np.concatenate((last_left_translation, np.array([0])), axis=0)
    # concat R and last_left_translation_homogenus to get 4x4 matrix
    last_left_transformation_homogenus = np.concatenate((R, last_left_translation_homogenus.reshape(4, 1)), axis=1)
    print(last_left_transformation_homogenus.shape)
    print(P_right.shape)
    last_left_transformation = np.hstack((last_left_rotation, last_left_translation))
    # P_right_homogens = np.vstack((P_right, np.array([0, 0, 0, 1])))
    last_right_transformation = compose_transformations(last_left_transformation, P_right)
    # convert to inverse by gtsam.Pose3.inverse
    Rt_inverse = (gtsam.Pose3.inverse(gtsam.Pose3(last_left_transformation_homogenus)))
    # last_left_translation_homogenus = (gtsam.Pose3.inverse(last_left_transformation_homogenus
    Rt_right_inverse = (gtsam.Pose3.inverse(gtsam.Pose3(last_right_transformation)))
    last_frame_camera_left = gtsam.StereoCamera(Rt_inverse, K)
    last_frame_camera_right = gtsam.StereoCamera(Rt_right_inverse, K)
    stereo_pt = gtsam.StereoPoint2(last_left_img_xy[0], last_right_img_xy[0], last_left_img_xy[1])
    meassurement = gtsam.Point2Vector([last_left_img_xy, last_right_img_xy])
    print(Rt_inverse, "\n")
    cameras = [last_frame_camera_left, last_frame_camera_right]
    # camera_set = gtsam.Cal3_S2Stereo(cameras)
    # p3d = gtsam.triangulatePoint3(camera_set, meassurement)
    #trying from note of moodle
    frame = gtsam.StereoCamera(Rt_inverse, K)
    p3d = frame.backproject(stereo_pt)
    print(p3d)
    # stereoPoint2 = frame.project(Point3)


def q1(db: TrackingDB):
    K, P_left, P_right = read_cameras_matrices()

    # Get all valid tracks
    valid_tracks = [track for track in db.all_tracks() if len(db.frames(track)) >= 10]
    # Select a random track of length >= 10
    selected_track = random.choice(valid_tracks)
    selected_track_frames = db.frames(selected_track)
    frame_ids = [frame_id for frame_id in db.frames(selected_track)]
    track_length = len(frame_ids)

    # Triangulate the 3D point using the last frame of the track
    link = db.link(selected_track_frames[-1], selected_track)
    # Read transformations
    transformations = read_poses_truth(seq=(frame_ids[0], frame_ids[-1] + 1))
    last_left_img_xy = link.left_keypoint()
    last_right_img_xy = link.right_keypoint()
    last_left_transformation = transformations[-1]
    last_l_projection_mat = K @ last_left_transformation
    last_r_projection_mat = K @ compose_transformations(last_left_transformation, P_right)
    p3d = xy_triangulation([last_left_img_xy, last_right_img_xy], last_l_projection_mat, last_r_projection_mat)

    left_projections = []
    right_projections = []

    for trans in transformations:
        left_proj_cam = K @ trans
        right_proj_cam = K @ compose_transformations(trans, P_right)

        left_proj = project(p3d, left_proj_cam)
        right_proj = project(p3d, right_proj_cam)

        left_projections.append(left_proj)
        right_projections.append(right_proj)

    frames_l_xy = [db.link(frame, selected_track).left_keypoint() for frame in selected_track_frames]
    frames_r_xy = [db.link(frame, selected_track).right_keypoint() for frame in selected_track_frames]
    left_proj_dist = get_euclidean_distance(np.array(left_projections), np.array(frames_l_xy))
    right_proj_dist = get_euclidean_distance(np.array(right_projections), np.array(frames_r_xy))

    # Reverse the order of the errors
    left_proj_dist = left_proj_dist[::-1]
    right_proj_dist = right_proj_dist[::-1]

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

    # 2. Use gtsam.StereoCamera with the global camera matrices
    stereo_cameras = []
    for i in range(track_length):
        R = db.rotation_matrices[frame_ids[i]]
        t = db.translation_vectors[frame_ids[i]]
        # K = db.K  # Assuming intrinsic matrix K is available in db
        # R = gtsam.
        stereo_cam = gtsam.StereoCamera(K, R, t)
        stereo_cameras.append(stereo_cam)

    # 3. Triangulate a 3D point in global coordinates from the last frame
    last_frame_id = frame_ids[-1]
    last_frame = db.frames[last_frame_id]
    stereo_camera_last = stereo_cameras[-1]
    left_image_point = last_frame.left_image_point  # Replace with actual point
    right_image_point = last_frame.right_image_point  # Replace with actual point
    triangulated_point = stereo_camera_last.triangulatePoint(gtsam.StereoPoint2(left_image_point, right_image_point))

    # 4. Project this point to all frames in the track
    reprojection_errors = []
    for i in range(track_length):
        stereo_camera = stereo_cameras[i]
        reprojected_point = stereo_camera.project(triangulated_point)
        left_error = np.linalg.norm(reprojected_point.left() - db.frames[frame_ids[i]].left_image_point)
        right_error = np.linalg.norm(reprojected_point.right() - db.frames[frame_ids[i]].right_image_point)
        reprojection_errors.append(np.linalg.norm([left_error, right_error]))

    # 5. Calculate and plot the reprojection error
    plt.figure()
    plt.plot(range(track_length), reprojection_errors, label='Reprojection Error')
    plt.xlabel('Frame Index')
    plt.ylabel('Reprojection Error (L2 Norm)')
    plt.title('Reprojection Error Over Track Frames')
    plt.legend()
    plt.show()

    # 6. Create factors and plot the factor error
    factor_errors = []
    noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)  # Example noise model
    for i in range(track_length):
        stereo_camera = stereo_cameras[i]
        reprojected_point = stereo_camera.project(triangulated_point)
        factor = gtsam.GenericProjectionFactorStereo(gtsam.StereoPoint2(db.frames[frame_ids[i]].left_image_point, db.frames[frame_ids[i]].right_image_point), noise_model, gtsam.symbol('x', i), gtsam.symbol('l', 0), K)
        factor_error = factor.error(gtsam.Values({gtsam.symbol('x', i): stereo_camera.pose(), gtsam.symbol('l', 0): triangulated_point}))
        factor_errors.append(factor_error)

    plt.figure()
    plt.plot(range(track_length), factor_errors, label='Factor Error')
    plt.xlabel('Frame Index')
    plt.ylabel('Factor Error')
    plt.title('Factor Error Over Track Frames')
    plt.legend()
    plt.show()

    # 7. Address covariance matrix and factor error as a function of reprojection error
    print("Covariance matrix used:", noise_model.covariance())
    correlation = np.corrcoef(reprojection_errors, factor_errors)[0, 1]
    print("Correlation between factor error and reprojection error:", correlation)


if __name__ == '__main__':
    # help(gtsam)
    db = TrackingDB()
    db.load('db')
    q1_avish(db)
    # q1(db)
    # K, P_left, P_right = read_cameras_matrices()
    # print(K)
    # print("0\n", db.rotation_matrices[0])
    # print("1\n", db.rotation_matrices[1])
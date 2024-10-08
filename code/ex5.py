import random
# import matplotlib
# matplotlib.use('TkAgg')
import gtsam
import numpy as np
from gtsam import symbol
import gtsam.utils.plot as gtsam_plot
from BundleWindow import BundleWindow
from BundleAdjustment import BundleAdjustment
from algorithms_library import (get_euclidean_distance, create_calib_mat_gtsam,
                                create_ext_matrix_gtsam, triangulate_gtsam,
                                find_projection_factor_with_largest_initial_error, print_projection_details,
                                read_poses, calculate_trajectory, calculate_trajectory_key_frames_gtsam,
                                plot_root_ground_truth_and_estimate, compose_transformations_gtsam,
                                gtsam_compose_to_first_kf, convert_rel_landmarks_to_global,
                                plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth, plot_cameras_path)
import matplotlib.pyplot as plt


from tracking_database import TrackingDB

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


def q1(db: TrackingDB):
    calib_mat_gtsam = create_calib_mat_gtsam()
    values = gtsam.Values()

    valid_tracks = [track for track in db.all_tracks() if len(db.frames(track)) >= 10]
    selected_track = random.choice(valid_tracks)
    selected_track_frames = db.frames(selected_track)
    frame_ids = [frame_id for frame_id in db.frames(selected_track)]
    track_length = len(frame_ids)

    # Triangulate the 3D point using the last frame of the track
    link = db.link(selected_track_frames[-1], selected_track)
    # Read transformations

    Rt_inverse_gtsam = create_ext_matrix_gtsam(db, frame_ids[-1])
    triangulate_p3d_gtsam = triangulate_gtsam(Rt_inverse_gtsam, calib_mat_gtsam, link)

    # Update values dictionary
    p3d_sym = symbol("q", 0)
    values.insert(p3d_sym, triangulate_p3d_gtsam)

    # calculate re-projection
    left_projections = []
    right_projections = []
    selected_track_frames = selected_track_frames[::-1]
    frame_ids = frame_ids[::-1]

    frames_l_xy = [db.link(frame, selected_track).left_keypoint() for frame in selected_track_frames]
    frames_r_xy = [db.link(frame, selected_track).right_keypoint() for frame in selected_track_frames]

    factors = []
    for i in range(track_length):
        # Read transformations
        current_left_rotation = db.rotation_matrices[frame_ids[i]]
        current_left_translation = db.translation_vectors[frame_ids[i]]

        # concat row of zeros to R to get 4x3 matrix
        R = np.concatenate((current_left_rotation, np.zeros((1, 3))), axis=0)

        current_left_translation = current_left_translation.reshape(3, 1)
        # concat last_left_translation_homogenus with zero to get 4x1 vector
        zero_row = np.array([0]).reshape(1, 1)
        current_left_translation_homogenus = np.vstack((current_left_translation, zero_row))

        # concat R and last_left_translation_homogenus to get 4x4 matrix
        current_left_transformation_homogenus = np.concatenate((R, current_left_translation_homogenus.reshape(4, 1)),
                                                               axis=1)
        #todo: maybe not all the row zero check it!!!

        # convert to inverse by gtsam.Pose3.inverse
        Rt_inverse_gtsam_current = (gtsam.Pose3.inverse(gtsam.Pose3(current_left_transformation_homogenus)))
        current_frame_camera_left = gtsam.StereoCamera(Rt_inverse_gtsam_current, calib_mat_gtsam)
        project_stereo_pt_gtsam = current_frame_camera_left.project(triangulate_p3d_gtsam)

        left_projections.append([project_stereo_pt_gtsam.uL(), project_stereo_pt_gtsam.v()])
        right_projections.append([project_stereo_pt_gtsam.uR(), project_stereo_pt_gtsam.v()])

        left_pose_sym = symbol("c", frame_ids[i])
        values.insert(left_pose_sym, Rt_inverse_gtsam_current)

        # Factor creation
        gtsam_measurement_pt2 = gtsam.StereoPoint2(frames_l_xy[i][0], frames_r_xy[i][0], frames_l_xy[i][1])
        projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                             symbol("c", frame_ids[i]), p3d_sym, calib_mat_gtsam)
        factors.append(factor)

    left_proj_dist = get_euclidean_distance(np.array(left_projections), np.array(frames_l_xy))
    right_proj_dist = get_euclidean_distance(np.array(right_projections), np.array(frames_r_xy))
    total_proj_dist = (left_proj_dist + right_proj_dist) / 2

    # Factor error
    factor_projection_errors = projection_factors_error(factors, values)

    # Plots re-projection error
    plot_re_projection_error(left_proj_dist, right_proj_dist, selected_track)
    plot_factor_error_graph(factor_projection_errors, -1)
    plot_factor_error_as_function_of_projection_error(total_proj_dist, factor_projection_errors)


def plot_re_projection_error(left_proj_dist, right_proj_dist, selected_track):
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


def plot_factor_error_graph(factor_projection_errors, frame_idx_triangulate):
    """
    Plots re projection error
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    frame_title = "Last" if frame_idx_triangulate == -1 else "First"
    ax.set_title(f"Factor error from {frame_title} frame")
    plt.scatter(range(len(factor_projection_errors)), factor_projection_errors, label="Factor")
    plt.legend(loc="upper right")
    plt.ylabel('Error')
    plt.xlabel('Frames')
    fig.savefig(f"Factor error graph for {frame_title} frame.png")
    plt.close(fig)


def plot_factor_error_as_function_of_projection_error(projection_error: np.array, factor_error: np.array,
                                                      title='Factor Error as a function of Re-Projection Error'):
    """
    Plots the factor error as a function of the projection error.

    :param projection_error: NumPy array of projection errors for each frame.
    :param factor_error: NumPy array of factor errors for each frame.
    :param title: Title of the plot.
    """
    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(projection_error, factor_error, color='b')
    # Add labels and title
    plt.xlabel('Projection Error')
    plt.ylabel('Factor Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig("Factor error as a function of a Re-projection error")


def plot_sphere(ax, center, radius=0.1, color='b'):
    # Create a sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=0.3)


def q3(db):
    # todo:maybe visualize the measurments and the projections in the image (bullet 3).
    first_window = BundleWindow(0, 20, db=db)
    first_window.create_factor_graph()
    error_before_optim = first_window.calculate_graph_error(False)
    first_window.optimize()
    error_after_optim = first_window.calculate_graph_error()
    # todo: check if there a way to reduce more error.

    print(f"The error of the graph before optimize:{error_before_optim}")
    print(f"The error of the graph after optimize:{error_after_optim}")
    print(f"The number of the factors in the graph: {first_window.graph.size()}")
    print(f"The average factor error before optimization:{error_before_optim / first_window.graph.size()}")
    print(f"The average factor error after optimization:{error_after_optim / first_window.graph.size()}")
    max_error_factor, max_error = find_projection_factor_with_largest_initial_error(first_window.graph,
                                                                                    first_window.get_initial_estimate())
    print_projection_details(max_error_factor, first_window.get_initial_estimate(), create_calib_mat_gtsam())
    print_projection_details(max_error_factor, first_window.get_optimized_values(), create_calib_mat_gtsam())

    gtsam.utils.plot.plot_trajectory(fignum=0, values=first_window.get_optimized_values(), title="aa")

    plt.savefig("optimized_trajectory.png")  # Save the plot to a file
    plt.close()  # Close the plot
    #check with (c, 0) and c(,latrframe)
    cams = first_window.get_optimized_cameras()
    cams_relative = []
    for t in cams:
        cams_relative.append(t.translation())
    cams_relative = np.array(cams_relative)

    plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth(cameras=cams_relative,
                                                                       landmarks=first_window.get_optimized_landmarks())


import gtsam
import numpy as np
import matplotlib.pyplot as plt
from gtsam import symbol


def q4(db):
    num_frames = 100
    truth_poses_R, truth_poses_t = read_poses(num_frames)
    bundle_adjustment = BundleAdjustment(db, 0, num_frames - 1)
    bundle_adjustment.solve_with_window_size_20()
    bundle_windows = bundle_adjustment.get_windows()
    all_optimized_values = bundle_adjustment.get_all_optimized_values()
    global_cams = []
    relative_cams = []
    landmarkd_x_cor = []
    landmarkd_y_cor = []
    landmarkd_z_cor = []
    windows_indices = []
    for i in range(0, num_frames,20):
        first_kf = i
        if (i < num_frames - 20):
            last_kf = i + 20
        else:
            last_kf = num_frames - 1
        windows_indices.append([first_kf, last_kf])


    for i, (window_idx, window_bundle) in enumerate(zip(windows_indices,bundle_windows)):
        cams = [window_bundle.get_optimized_cameras()]

        # cams = [window_bundle.get_optimized_values().atPose3(symbol('c', frame_id)) for frame_id in range(window_idx[0], window_idx[1] + 1)]
        if i == 0:
            global_cams.extend(cams)
        else:
            to_global_cor = global_cams[-1]
            global_cams.extend([to_global_cor.compose(cam) for cam in cams])

        for lm in window_bundle.get_landmarks_symbols_set():
            landmark = window_bundle.get_optimized_values().atPoint3(lm)
            global_landmark = global_cams[-1].transformFrom(landmark)
            landmarkd_x_cor.append(global_landmark[0])
            landmarkd_y_cor.append(global_landmark[1])
            landmarkd_z_cor.append(global_landmark[2])
        if i == len(windows_indices) - 1:
            first_frame_cam = cams[0]
            print(f"Position of the first frame in the last bundle: {first_frame_cam.translation()}")
            #calculate anchoring erroe:


    landmarkd_x_cor = np.array(landmarkd_x_cor)
    landmarkd_y_cor = np.array(landmarkd_y_cor)
    landmarkd_z_cor = np.array(landmarkd_z_cor)
    landmarks_cors = np.vstack((landmarkd_x_cor, landmarkd_y_cor, landmarkd_z_cor)).T
    x_cors = [cam.x() for cam in global_cams]
    y_cors = [cam.y() for cam in global_cams]
    z_cors = [cam.z() for cam in global_cams]
    cams_cors = np.vstack((x_cors, y_cors, z_cors)).T
    plot_cameras_path(cams_cors,landmarks_cors)
    return





    # # Print the position of the first frame in the last bundle
    # last_window = bundle_adjustment.get_last_window()
    # last_window_range = last_window.get_frame_range()
    # last_window_first_frame_key = symbol('c', 0)
    # if all_optimized_values[-1].exists(last_window_first_frame_key):
    #     last_window_first_frame_pose = all_optimized_values[-1].atPose3(last_window_first_frame_key)
    #     print(f"Position of the first frame in the last bundle: {last_window_first_frame_pose.translation()}")
    # else:
    #     print("The first frame of the last window is not in the optimized values.")
    #
    # # Print the anchoring factor final error
    # if last_window.graph.size() > 0:
    #     anchoring_factor = last_window.graph.at(0)
    #     anchoring_error = anchoring_factor.error(last_window.get_optimized_values())
    #     print(f"Anchoring factor final error: {anchoring_error}")
    # else:
    #     print("No anchoring factor found in the last window.")

    #extract all 3d points of the keyframes
    landmarks = []
    for window in bundle_windows:
        window_landmarks = []
        for key in window.get_landmarks_symbols_set():
            landmark = window.get_optimized_values().atPoint3(key)
            window_landmarks.append(landmark)
        landmarks.append(window_landmarks)


    # Extract poses for all keyframes
    keyframe_poses = []
    for i, window_values in enumerate(all_optimized_values):
        keyframe_poses.append(window_values.atPose3(symbol('c', i * 20)))



    # print(keyframe_poses[0])
    # print(keyframe_poses[0].translation())
    # for i in range(0, NUM_FRAMES, 20):
    #     key = symbol('c', i)
    #     if all_optimized_values.exists(key):
    #         keyframe_poses.append(all_optimized_values.atPose3(key))



    truth_poses_kf_R = [truth_poses_R[i] for i in range(0, num_frames, 20)]
    truth_poses_kf_t = [truth_poses_t[i] for i in range(0, num_frames, 20)]
    ground_truth_trajectory = calculate_trajectory(truth_poses_kf_R, truth_poses_kf_t)
    #calculate poses relative to first_camera_coordinate_syste
    keyframe_poses_relative = gtsam_compose_to_first_kf(keyframe_poses)
    cameras = []
    cameras1 = []
    first_cam_truth = truth_poses_R[0]
    first_cam_truth_t = truth_poses_t[0]
    zero_row = np.array([0]).reshape(1, 1)
    first_cam_truth_t_homogenus = np.vstack((first_cam_truth_t, zero_row))
    first_cam_truth_homogenus = np.concatenate((first_cam_truth, np.zeros((1, 3))), axis=0)
    for window in bundle_windows:
        camera_relate_to_frame0 = window.covnert_last_kf_to_frame0_coordinates()
        # cam_R = window_cams[-1].rotation().matrix()
        # cam_t = window_cams[-1].translation()
        # cam_R_homogenus = np.concatenate((cam_R, np.zeros((1, 3))), axis=0)
        #
        # cam_t_homogenus = np.concatenate((cam_t.reshape(3, 1), zero_row), axis=0)
        # compose_R = first_cam_truth_homogenus.T @ cam_R_homogenus
        # temp = cam_t_homogenus - first_cam_truth_t_homogenus
        # compose_t = first_cam_truth_homogenus.T @ temp
        # compose_Rt = np.concatenate((compose_R, compose_t), axis=1)
        # #try1:
        # cam_relate = gtsam.Pose3(compose_Rt)
        cameras.append(gtsam.Pose3(camera_relate_to_frame0).translation())
        #
        #try2:
        cameras1.append((-camera_relate_to_frame0[0:3, 0:3].T @ camera_relate_to_frame0[0:3, 3]).reshape(3, 1))

    # cpnvert to numpy
    # keyframe_poses_relative = [keyframe_poses[0]]
    # for pose in keyframe_poses[1:]:
    #     keyframe_poses_relative.append(compose_transformations_gtsam(keyframe_poses[0], pose))
    # keyframe_poses_relative.append(compose_transformations(keyframe_poses[0], pose) for pose in keyframe_poses[1:])
    # global_landmarks = convert_rel_landmarks_to_global(cameras, landmarks)
    # cameras = np.array(cameras)
    #test for landmark non-rleative
    landmarks_flatten = []
    for window_landmarks in landmarks:
        for landmark in window_landmarks:
            landmarks_flatten.append(landmark)


    # for cam in cameras:


    plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth(cameras=np.array(cameras), landmarks=np.array(landmarks_flatten))

    # estimate_trajectory = calculate_trajectory_key_frames_gtsam(keyframe_poses_relative)
    initial_cameras = [window.get_initial_estimate_cameras()[-1] for window in bundle_windows]
    plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth(cameras=np.array(cameras1), landmarks=np.array(landmarks_flatten))


    # # Calculate and plot keyframe localization error
    # errors = []
    # for est, gt in zip(keyframe_poses_relative, ground_truth_trajectory ):
    #     est_trans = np.array([est.translation().x(), est.translation().y(), est.translation().z()])
    #     gt_trans = gt[:3, 3]
    #     error = np.linalg.norm(est_trans - gt_trans)
    #     errors.append(error)
    #
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(0, len(errors) * 20, 20), errors)
    # plt.title('Keyframe Localization Error Over Time')
    # plt.xlabel('Frame')
    # plt.ylabel('Error (meters)')
    # plt.grid(True)
    # plt.show()

if __name__ == '__main__':
    db = TrackingDB()
    db.load('db')
    # q1(db)
    # q3(db)
    q4(db)
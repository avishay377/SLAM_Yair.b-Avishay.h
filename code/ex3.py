import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
from algorithms_library import plot_camera_positions, plot_root_ground_truth_and_estimate, \
    plot_supporters_non_supporters, read_cameras_matrices, extract_keypoints_and_inliers, cv_triangulate_matched_points, \
    find_consensus_matches_indices, calculate_front_camera_matrix, extract_actual_consensus_pixels, \
    find_supporter_indices_for_model, \
    estimate_projection_matrices_with_ransac, \
    compute_trajectory_and_distance, plot_inliers_outliers_ransac, plot_two_3D_point_clouds, read_images_from_dataset
import cProfile
import pstats
import io


DATASET_PATH = os.path.join(os.getcwd(), r'dataset\sequences\00')
DETECTOR = cv2.SIFT_create()
# DEFAULT_MATCHER = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
MATCHER = cv2.FlannBasedMatcher(indexParams=dict(algorithm=0, trees=5),
                                searchParams=dict(checks=50))
NUM_FRAMES = 20
MAX_DEVIATION = 2
Epsilon = 1e-10


# def find_consensus_matches_indices(back_inliers, front_inliers, tracking_inliers):
#     # TODO: make this more efficient (from o(n^2) to O(n*logn)), see:
#     #  https://stackoverflow.com/questions/71764536/most-efficient-way-to-match-2d-coordinates-in-python
#     #   Also maybe filter tracking inliers (by the filtered descriptors and not the fully ones)
#     # Returns a list of 3-tuples indices, representing the idx of the consensus match
#     # in each of the 3 original match-lists
#     consensus = []
#     back_inliers_left_idx = [m.queryIdx for m in back_inliers]
#     front_inliers_left_idx = [m.queryIdx for m in front_inliers]
#     for idx in range(len(tracking_inliers)):
#         back_left_kp_idx = tracking_inliers[idx].queryIdx
#         front_left_kp_idx = tracking_inliers[idx].trainIdx
#         try:
#             idx_of_back_left_kp_idx = back_inliers_left_idx.index(back_left_kp_idx)
#             idx_of_front_left_kp_idx = front_inliers_left_idx.index(front_left_kp_idx)
#         except ValueError:
#             continue
#         consensus.append(tuple([idx_of_back_left_kp_idx, idx_of_front_left_kp_idx, idx]))
#     return consensus


def main():
    print("start session:\n")
    img0_left, img0_right = read_images_from_dataset(0)
    img1_left, img1_right = read_images_from_dataset(1)
    K, Ext0_left, Ext0_right = read_cameras_matrices()  # intrinsic & extrinsic camera Matrices
    R0_left, t0_left = Ext0_left[:, :3], Ext0_left[:, 3:]
    R0_right, t0_right = Ext0_right[:, :3], Ext0_right[:, 3:]

    # QUESTION 1
    q1_output = q1(K, R0_left, R0_right, img0_left, img0_right, img1_left, img1_right, t0_left, t0_right)
    (descriptors0_left, descriptors1_left, inliers_0_0, inliers_1_1, keypoints0_left, keypoints0_right, keypoints1_left,
     keypoints1_right, point_cloud_0) = q1_output
    # QUESTION 2
    tracking_matches = q2(descriptors0_left, descriptors1_left)
    # QUESTION 3
    q3_output = q3(Ext0_left, Ext0_right, K, R0_right, inliers_0_0, inliers_1_1, keypoints1_left, point_cloud_0,
                   t0_right, tracking_matches)
    R1_left, R1_right, consensus_match_indices_0_1, t1_left, t1_right = q3_output
    # QUESTION 4
    q4(K, R0_left, R0_right, R1_left, R1_right, consensus_match_indices_0_1, img0_left, img1_left, inliers_0_0,
       inliers_1_1, keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0, t0_left,
       t0_right, t1_left, t1_right, tracking_matches)
    # QUESTION 5:
    q5(K, R0_left, R0_right, consensus_match_indices_0_1, img0_left, img1_left, inliers_0_0, inliers_1_1,
       keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0, t0_left, t0_right,
       tracking_matches)
    # Question 6:
    NUM_FRAMES = 3360  # total number of stereo-images in our KITTI dataset
    q6(NUM_FRAMES)


def q6(NUM_FRAMES):
    estimated_trajectory, ground_truth_trajectory, distances = compute_trajectory_and_distance(num_frames=NUM_FRAMES,
                                                                                               verbose=True)
    plot_root_ground_truth_and_estimate(estimated_trajectory, ground_truth_trajectory)
    fig, axes = plt.subplots(1, 2)
    fig.suptitle('KITTI Trajectories')
    n = estimated_trajectory.T[0].shape[0]
    markers_sizes = np.ones((n,))
    markers_sizes[[i for i in range(n) if i % 50 == 0]] = 15
    markers_sizes[0], markers_sizes[-1] = 50, 50
    axes[0].scatter(estimated_trajectory.T[0], estimated_trajectory.T[2],
                    marker="o", s=markers_sizes, c=estimated_trajectory.T[1], cmap="Reds", label="estimated")
    axes[0].scatter(ground_truth_trajectory.T[0], ground_truth_trajectory.T[2],
                    marker="x", s=markers_sizes, c=ground_truth_trajectory.T[1], cmap="Greens", label="ground truth")
    axes[0].set_title("Trajectories")
    axes[0].legend(loc='best')

    axes[1].scatter([i for i in range(n)], distances, c='k', marker='*', s=1)
    axes[1].set_title("Euclidean Distance between Trajectories")
    fig.set_figwidth(10)
    plt.show()


def q5(K, R0_left, R0_right, consensus_match_indices_0_1, img0_left, img1_left, inliers_0_0, inliers_1_1,
       keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0, t0_left, t0_right,
       tracking_matches):
    mR, mt, sup = estimate_projection_matrices_with_ransac(point_cloud_0, consensus_match_indices_0_1,
                                                           inliers_0_0, inliers_1_1,
                                                           keypoints0_left, keypoints0_right,
                                                           keypoints1_left, keypoints1_right,
                                                           K, R0_left, t0_left, R0_right, t0_right,
                                                           verbose=True)
    plot_two_3D_point_clouds(mR, mt, point_cloud_0)
    plot_inliers_outliers_ransac(consensus_match_indices_0_1, img0_left, img1_left, keypoints0_left, keypoints1_left,
                                 sup, tracking_matches)


def q4(K, R0_left, R0_right, R1_left, R1_right, consensus_match_indices_0_1, img0_left, img1_left, inliers_0_0,
       inliers_1_1, keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0, t0_left,
       t0_right, t1_left, t1_right, tracking_matches):
    real_pixels = extract_actual_consensus_pixels(consensus_match_indices_0_1, inliers_0_0, inliers_1_1,
                                                  keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right)
    consensus_3d_points = point_cloud_0[[m[0] for m in consensus_match_indices_0_1]]
    Rs = [R0_left, R0_right, R1_left, R1_right]
    ts = [t0_left, t0_right, t1_left, t1_right]
    supporter_indices = find_supporter_indices_for_model(consensus_3d_points, real_pixels, K, Rs, ts)
    supporters = [consensus_match_indices_0_1[idx] for idx in supporter_indices]

    # plot the supporters on img0_left and img1_left:
    supporting_tracking_matches = [tracking_matches[idx] for (_, _, idx) in supporters]
    non_supporting_tracking_matches = [m for m in tracking_matches if m not in supporting_tracking_matches]
    supporting_pixels_back = [keypoints0_left[i].pt for i in [m.queryIdx for m in supporting_tracking_matches]]
    supporting_pixels_front = [keypoints1_left[i].pt for i in [m.trainIdx for m in supporting_tracking_matches]]
    non_supporting_pixels_back = [keypoints0_left[i].pt for i in [m.queryIdx for m in non_supporting_tracking_matches]]
    non_supporting_pixels_front = [keypoints1_left[i].pt for i in [m.trainIdx for m in non_supporting_tracking_matches]]
    plot_supporters_non_supporters(img0_left, img1_left, supporting_pixels_back, supporting_pixels_front,
                                   non_supporting_pixels_back, non_supporting_pixels_front)


def q3(Ext0_left, Ext0_right, K, R0_right, inliers_0_0, inliers_1_1, keypoints1_left, point_cloud_0, t0_right,
       tracking_matches):
    consensus_match_indices_0_1 = find_consensus_matches_indices(inliers_0_0, inliers_1_1, tracking_matches)
    is_success, R1_left, t1_left = calculate_front_camera_matrix(random.sample(consensus_match_indices_0_1, 4),
                                                                 point_cloud_0, inliers_1_1, keypoints1_left, K)

    Ext1_left = np.hstack((R1_left, t1_left))
    R1_right = np.dot(Ext1_left[:, :3], R0_right)
    t1_right = np.dot(Ext1_left[:, :3], t0_right) + t1_left
    Ext1_right = np.hstack((R1_right, t1_right.reshape(-1, 1)))
    plot_camera_positions([Ext0_left, Ext0_right, Ext1_left, Ext1_right])
    return R1_left, R1_right, consensus_match_indices_0_1, t1_left, t1_right


def q2(descriptors0_left, descriptors1_left):
    # find matches in the first tracking pair (img0, img1) and sort for consensus-match
    tracking_matches = sorted(MATCHER.match(descriptors0_left, descriptors1_left), key=lambda m: m.queryIdx)
    return tracking_matches


def q1(K, R0_left, R0_right, img0_left, img0_right, img1_left, img1_right, t0_left, t0_right):
    # triangulate keypoints from stereo pair 0:
    preprocess_pair_0_0 = extract_keypoints_and_inliers(img0_left, img0_right)
    keypoints0_left, descriptors0_left, keypoints0_right, descriptors0_right, inliers_0_0, _ = preprocess_pair_0_0
    point_cloud_0 = cv_triangulate_matched_points(keypoints0_left, keypoints0_right, inliers_0_0,
                                                  K, R0_left, t0_left, R0_right, t0_right)
    # triangulate keypoints from stereo pair 1:
    preprocess_pair_1_1 = extract_keypoints_and_inliers(img1_left, img1_right)
    keypoints1_left, descriptors1_left, keypoints1_right, descriptors1_right, inliers_1_1, _ = preprocess_pair_1_1
    # triangulate pair_1 inliers based on projection matrices from pair_0
    point_cloud_1_with_camera_0 = cv_triangulate_matched_points(keypoints1_left, keypoints1_right, inliers_1_1,
                                                                K, R0_left, t0_left, R0_right, t0_right)
    return descriptors0_left, descriptors1_left, inliers_0_0, inliers_1_1, keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0


if __name__ == '__main__':
    ANALYZE = False
    if ANALYZE:
        # Profile the code
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()

        # Create a stream to hold the profile data
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
        stats.print_stats()

        # Print the profiling results
        print(stream.getvalue())
    else:
        main()

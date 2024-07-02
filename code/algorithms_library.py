import os
import random
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib



DATASET_PATH = os.path.join(os.getcwd(), r'dataset\sequences\00')
DETECTOR = cv2.SIFT_create()
# DEFAULT_MATCHER = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
MATCHER = cv2.FlannBasedMatcher(indexParams=dict(algorithm=0, trees=5),
                                searchParams=dict(checks=50))
NUM_FRAMES = 20
MAX_DEVIATION = 2
Epsilon = 1e-10


matplotlib.use('TkAgg')

DATA_PATH = '../../VAN_ex/dataset/sequences/00/'



def detect_keypoints(img, method='ORB', num_keypoints=500):
    """
    Detects keypoints in an image using the specified method.

    Args:
    - img (np.array): Input image in which to detect keypoints.
    - method (str): Feature detection method ('ORB', 'AKAZE', 'SIFT').
    - num_keypoints (int): Number of keypoints to detect.

    Returns:
    - keypoints (list): Detected keypoints.
    - descriptors (np.array): Descriptors of the detected keypoints.
    """
    if method == 'ORB':
        detector = cv2.ORB_create(nfeatures=num_keypoints)
    elif method == 'AKAZE':
        detector = cv2.AKAZE_create()
    elif method == 'SIFT':
        detector = cv2.SIFT_create()
    else:
        raise ValueError(f"Unsupported method: {method}")

    keypoints, descriptors = detector.detectAndCompute(img, None)
    return keypoints, descriptors


def get_matches_from_kpts(kp1, kp2):
    pass


def draw_keypoints(img, keypoints):
    """
    Draws keypoints on an image.

    Args:
    - img (np.array): Input image.
    - keypoints (list): Detected keypoints.

    Returns:
    - img_with_keypoints (np.array): Image with keypoints drawn.
    """
    return cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)


def read_images(idx):
    """
    Reads a pair of stereo images from the dataset.

    Args:
    - idx (int): Index of the image pair.

    Returns:
    - img1 (np.array): First image of the stereo pair.
    - img2 (np.array): Second image of the stereo pair.
    """
    img_name = '{:06d}.png'.format(idx)

    img1 = cv2.imread(DATASET_PATH + f'image_0/' + img_name, 0)
    img2 = cv2.imread(DATASET_PATH + f'image_1/' + img_name, 0)
    return img1, img2


def apply_ratio_test(matches, ratio_threshold=0.5):
    """
    Applies the ratio test to reject matches.

    Args:
    - matches (list): List of matches obtained from matching descriptors.
    - ratio_threshold (float): Threshold value for the ratio of distances to reject matches.

    Returns:
    - good_matches (list): List of matches passing the ratio test.
    """
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    return good_matches


def match_keypoints(descriptors1, descriptors2):
    """
        Matches keypoints between two sets of descriptors using the Brute Force Matcher with Hamming distance.

        Args:
        - descriptors1 (np.array): Descriptors of keypoints from the first image.
        - descriptors2 (np.array): Descriptors of keypoints from the second image.

        Returns:
        - matches (list): List of matches between keypoints in the two images.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return matches


def read_cameras(calib_file):
    """
       Reads the camera calibration file and extracts the intrinsic and extrinsic parameters.

       Args:
       - calib_file (str): Path to the camera calibration file.

       Returns:
       - k (np.array): Intrinsic camera matrix (3x3).
       - m1 (np.array): Extrinsic parameters of the first camera (3x4).
       - m2 (np.array): Extrinsic parameters of the second camera (3x4).
    """
    with open(calib_file) as f:
        l1 = f.readline().split()[1:]  # Skip first token
        l2 = f.readline().split()[1:]  # Skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


def triangulation_process(P0, P1, inliers, k, keypoints1, keypoints2, plot=True):
    """
        Performs the triangulation process for a set of inlier matches between two images and plots the 3D points.

        Args:
        - P0 (np.array): Projection matrix of the first camera (3x4).
        - P1 (np.array): Projection matrix of the second camera (3x4).
        - inliers (list): List of inlier matches.
        - k (np.array): Intrinsic camera matrix (3x3).
        - keypoints1 (list): List of keypoints in the first image.
        - keypoints2 (list): List of keypoints in the second image.

        Returns:
        - points_3D_custom (np.array): Array of triangulated 3D points.
        - pts1 (np.array): Array of inlier points from the first image.
        - pts2 (np.array): Array of inlier points from the second image.
    """

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in inliers]).T
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in inliers]).T
    points_3D_custom = triangulation(k @ P0, k @ P1, pts1.T, pts2.T)
    # Example usage
    # points = np.random.rand(100, 3) * 10  # Generate some random 3D points
    # plot_3d_points(points, title="3D Points Example", xlim=(0, 10), ylim=(0, 10), zlim=(0, 10))
    if plot:
        plot_3d_points(points_3D_custom, title="Custom Triangulation", xlim=(-10, 10), ylim=(-10, 10), zlim=(-20, 150))
    return points_3D_custom, pts1, pts2


def linear_least_square_pts(left_cam_matrix, right_cam_matrix, left_kp, right_kp):
    """
        Computes the 3D point using linear least squares from corresponding 2D points in stereo images.
        Args:
        - left_cam_matrix (np.array): Projection matrix of the left camera (3x4).
        - right_cam_matrix (np.array): Projection matrix of the right camera (3x4).
        - left_kp (tuple): 2D point in the left image.
        - right_kp (tuple): 2D point in the right image.

        Returns:
        - np.array: 4D homogeneous coordinates of the triangulated point.
    """
    mat_a = np.array([left_cam_matrix[2] * left_kp[0] - left_cam_matrix[0],
                      left_cam_matrix[2] * left_kp[1] - left_cam_matrix[1],
                      right_cam_matrix[2] * right_kp[0] - right_cam_matrix[0],
                      right_cam_matrix[2] * right_kp[1] - right_cam_matrix[1]])
    _, _, vT = np.linalg.svd(mat_a)
    return vT[-1]


def triangulation(left_cam_matrix, right_cam_matrix, left_kp_list, right_kp_list):
    """
        Triangulates 3D points from corresponding 2D points in stereo images.

        Args:
        - left_cam_matrix (np.array): Projection matrix of the left camera (3x4).
        - right_cam_matrix (np.array): Projection matrix of the right camera (3x4).
        - left_kp_list (list): List of 2D points in the left image.
        - right_kp_list (list): List of 2D points in the right image.

        Returns:
        - np.array: Array of triangulated 3D points.
    """
    num_kp = len(left_kp_list)
    triangulation_pts = []
    for i in range(num_kp):
        p4d = linear_least_square_pts(left_cam_matrix, right_cam_matrix, left_kp_list[i], right_kp_list[i])
        p3d = p4d[:3] / p4d[3]
        triangulation_pts.append(p3d)
    return np.array(triangulation_pts)


def plot_3d_points(points, title="3D Points", xlim=None, ylim=None, zlim=None):
    """
    Plots 3D points using matplotlib with fixed axis limits.

    Args:
    - points (np.array): Array of 3D points.
    - title (str): Title of the plot (default is "3D Points").
    - xlim (tuple): Limits for the x-axis (min, max).
    - ylim (tuple): Limits for the y-axis (min, max).
    - zlim (tuple): Limits for the z-axis (min, max).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)

    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    plt.show()


def get_stereo_matches_with_filtered_keypoints_avish_test(img_left, img_right, feature_detector='AKAZE',
                                                          max_deviation=2):
    """
    Performs stereo matching with filtered keypoints between two images.

    Args:
    - img_left (np.array): Left image.
    - img_right (np.array): Right image.
    - feature_detector (str): Feature detector to use ('ORB', 'AKAZE'). Default is 'AKAZE'.
    - max_deviation (int): Maximum vertical deviation threshold for matching keypoints. Default is 2 pixels.

    Returns:
    - filtered_keypoints_left (list): Filtered keypoints from the left image.
    - filtered_keypoints_right (list): Filtered keypoints from the right image.
    - filtered_descriptors_left (np.array): Descriptors corresponding to filtered keypoints in the left image.
    - filtered_descriptors_right (np.array): Descriptors corresponding to filtered keypoints in the right image.
    - good_matches (list): List of good matches passing the deviation threshold.
    - keypoints_left (list): All keypoints detected in the left image.
    - keypoints_right (list): All keypoints detected in the right image.
    """
    # Initialize the feature detector
    if feature_detector == 'ORB':
        detector = cv2.ORB_create()
    elif feature_detector == 'AKAZE':
        detector = cv2.AKAZE_create(threshold=0.001, nOctaveLayers=2)
    else:
        raise ValueError("Unsupported feature detector")
    # Detect keypoints and compute descriptors
    keypoints_left, descriptors_left = detector.detectAndCompute(img_left, None)
    keypoints_right, descriptors_right = detector.detectAndCompute(img_right, None)
    # # Initialize the matcher
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if feature_detector == 'ORB' else (cv2.BFMatcher
    #                                                                                          (cv2.NORM_L2,
    #                                                                                           crossCheck=True))
    bf = cv2.BFMatcher()
    # Match descriptors
    matches = bf.match(descriptors_left, descriptors_right)
    # Filter matches based on the deviation threshold
    filtered_keypoints_left = []
    filtered_keypoints_right = []
    filtered_descriptors_left = []
    filtered_descriptors_right = []
    good_matches = []
    i = 0
    for match in matches:
        pt_left = keypoints_left[match.queryIdx].pt
        pt_right = keypoints_right[match.trainIdx].pt
        if abs(pt_left[1] - pt_right[1]) <= max_deviation:
            filtered_keypoints_left.append(keypoints_left[match.queryIdx])
            filtered_keypoints_right.append(keypoints_right[match.trainIdx])
            filtered_descriptors_left.append(descriptors_left[match.queryIdx])
            filtered_descriptors_right.append(descriptors_right[match.trainIdx])
            # maybe we can do as follows:
            match.trainIdx = i
            match.queryIdx = i
            good_matches.append(match)
            i += 1
    filtered_descriptors_left = np.array(filtered_descriptors_left)
    filtered_descriptors_right = np.array(filtered_descriptors_right)

    return filtered_keypoints_left, filtered_keypoints_right, filtered_descriptors_left, filtered_descriptors_right, good_matches, keypoints_left, keypoints_right


def plot_supporters_non_supporters(img0_left, img1_left, supporting_pixels_back, supporting_pixels_front,
                                   non_supporting_pixels_back, non_supporting_pixels_front):
    """
    Plots keypoints classified as supporters and non-supporters in two images.

    Args:
    - img0_left (np.array): Left image 0.
    - img1_left (np.array): Left image 1.
    - supporting_pixels_back (list): List of supporting keypoints in the back image.
    - supporting_pixels_front (list): List of supporting keypoints in the front image.
    - non_supporting_pixels_back (list): List of non-supporting keypoints in the back image.
    - non_supporting_pixels_front (list): List of non-supporting keypoints in the front image.
    """
    # Create a figure to hold both subplots
    fig, ax = plt.subplots(2, 1, figsize=(6, 12))
    # Plotting image left0
    ax[0].imshow(img0_left, cmap='gray')
    ax[0].set_title("Left Image 0")
    ax[0].axis('off')  # Turn off the axis
    for pt in supporting_pixels_back:
        ax[0].plot(pt[0], pt[1], 'o', color='cyan', markersize=1)  # Smaller points
    for pt in non_supporting_pixels_back:
        ax[0].plot(pt[0], pt[1], 'o', color='red', markersize=1)  # Smaller points
    # Plotting image left1
    ax[1].imshow(img1_left, cmap='gray')
    ax[1].set_title("Left Image 1")
    ax[1].axis('off')  # Turn off the axis
    for pt in supporting_pixels_front:
        ax[1].plot(pt[0], pt[1], 'o', color='cyan', markersize=1)  # Smaller points
    for pt in non_supporting_pixels_front:
        ax[1].plot(pt[0], pt[1], 'o', color='red', markersize=1)  # Smaller points

    # Finalizing plot settings
    plt.suptitle("q4 - supporters and unsupporters after pnp")
    plt.tight_layout()  # Adjust subplots to give some space
    plt.show()


def get_stereo_matches_with_filtered_keypoints(img_left, img_right, feature_detector='AKAZE', max_deviation=2):
    """
    Performs stereo matching with filtered keypoints between two images using pre-defined detector and matcher.

    Args:
    - img_left (np.array): Left image.
    - img_right (np.array): Right image.
    - feature_detector (str): Feature detector to use ('ORB', 'AKAZE'). Default is 'AKAZE'.
    - max_deviation (int): Maximum vertical deviation threshold for matching keypoints. Default is 2 pixels.

    Returns:
    - filtered_keypoints_left (list): Filtered keypoints from the left image.
    - filtered_keypoints_right (list): Filtered keypoints from the right image.
    - filtered_descriptors_left (np.array): Descriptors corresponding to filtered keypoints in the left image.
    - filtered_descriptors_right (np.array): Descriptors corresponding to filtered keypoints in the right image.
    - good_matches (list): List of good matches passing the deviation threshold.
    - keypoints_left (list): All keypoints detected in the left image.
    - keypoints_right (list): All keypoints detected in the right image.
    """

    # Initialize the feature detector
    if feature_detector == 'ORB':
        detector = cv2.ORB_create()
    elif feature_detector == 'AKAZE':
        detector = cv2.AKAZE_create(threshold=0.001, nOctaveLayers=2)
    else:
        raise ValueError("Unsupported feature detector")

    # Detect keypoints and compute descriptors
    keypoints_left, descriptors_left = DETECTOR.detectAndCompute(img_left, None)
    keypoints_right, descriptors_right = DETECTOR.detectAndCompute(img_right, None)
    bf = MATCHER
    # Match descriptors
    matches = bf.match(descriptors_left, descriptors_right)

    # Filter matches based on the deviation threshold
    filtered_keypoints_left = []
    filtered_keypoints_right = []
    filtered_descriptors_left = []
    filtered_descriptors_right = []
    good_matches = []

    for match in matches:
        pt_left = keypoints_left[match.queryIdx].pt
        pt_right = keypoints_right[match.trainIdx].pt
        if abs(pt_left[1] - pt_right[1]) <= max_deviation:
            filtered_keypoints_left.append(keypoints_left[match.queryIdx])
            filtered_keypoints_right.append(keypoints_right[match.trainIdx])
            filtered_descriptors_left.append(descriptors_left[match.queryIdx])
            filtered_descriptors_right.append(descriptors_right[match.trainIdx])
            # maybe we can do as follows:
            # match.trainIdx = i (when i is the number of iteration)
            # match.queryIdx = i
            good_matches.append(match)

    filtered_descriptors_left = np.array(filtered_descriptors_left)
    filtered_descriptors_right = np.array(filtered_descriptors_right)

    return filtered_keypoints_left, filtered_keypoints_right, filtered_descriptors_left, filtered_descriptors_right, good_matches, keypoints_left, keypoints_right


def plot_root_ground_truth_and_estimate(estimated_locations, ground_truth_locations):
    """
    Plots the camera trajectory based on estimated and ground truth locations.

    Args:
    - estimated_locations (np.array): Estimated trajectory locations.
    - ground_truth_locations (np.array): Ground truth trajectory locations.
    """

    # Plot the trajectories
    plt.figure(figsize=(10, 8))
    plt.plot(ground_truth_locations[:, 0], ground_truth_locations[:, 2], label='Ground Truth', color='r',
             linestyle='--')
    plt.plot(estimated_locations[:, 0], estimated_locations[:, 2], label='Estimated', color='b', marker='o')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Camera Trajectory')
    plt.legend()
    plt.grid(True)
    plt.show()


# def plot_3d_points(points, title="3D Points"):
#     """
#         Plots 3D points using matplotlib.
#
#         Args:
#         - points (np.array): Array of 3D points.
#         - title (str): Title of the plot (default is "3D Points").
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.title(title)
#     plt.show()


def reject_matches(keypoints1, keypoints2, matches):
    """
    Rejects matches based on vertical deviation between corresponding keypoints.

    Args:
    - keypoints1 (list): List of keypoints in the first image.
    - keypoints2 (list): List of keypoints in the second image.
    - matches (list): List of matches between keypoints.

    Returns:
    - deviations (list): List of vertical deviations for each match.
    - inliers (list): List of matches classified as inliers (deviation <= 2 pixels).
    - outliers (list): List of matches classified as outliers (deviation > 2 pixels).
    - indices (dict): Dictionary mapping keypoints from the first image to corresponding keypoints in the second image.
    """
    deviations = []
    inliers = []
    outliers = []
    indices = {}
    # idx = 0
    kp_mathces_im1 = [match.queryIdx for match in matches]
    kp_mathces_im2 = [match.trainIdx for match in matches]
    for i, j in zip(kp_mathces_im1, kp_mathces_im2):
        if abs(keypoints1[i].pt[1] - keypoints2[j].pt[1]) <= 2:
            indices[i] = j

    for i, match in enumerate(matches):
        # for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        deviation = abs(pt1[1] - pt2[1])  # Vertical deviation
        deviations.append(deviation)
        if deviation > 2:
            outliers.append(match)
        else:
            inliers.append(match)
    return deviations, inliers, outliers, indices


def reject_matches_and_remove_keypoints(keypoints1, keypoints2, matches):
    """
    Rejects matches based on vertical deviation between corresponding points.
plot
    Args:
    - keypoints1 (list): List of keypoints in the first image.
    - keypoints2 (list): List of keypoints in the second image.
    - matches (list): List of matches between keypoints.

    Returns:
    - deviations (list): List of vertical deviations.
    - inliers (list): List of matches with deviations <= 2 pixels.
    - outliers (list): List of matches with deviations > 2 pixels.
    """
    deviations = []
    inliers = []
    outliers = []

    # Create copies of keypoints lists to avoid modifying the originals
    # Convert keypoints1 and keypoints2 to lists if they are tuples
    if isinstance(keypoints1, tuple):
        keypoints1 = list(keypoints1)
    if isinstance(keypoints2, tuple):
        keypoints2 = list(keypoints2)
    keypoints1_filtered = keypoints1.copy()
    keypoints2_filtered = keypoints2.copy()

    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        deviation = abs(pt1[1] - pt2[1])  # Vertical deviation
        deviations.append(deviation)
        deviations.append(deviation)

        if deviation > 2:
            # Remove keypoints from filtered lists
            keypoints1_filtered[match.queryIdx] = None
            keypoints2_filtered[match.trainIdx] = None
            outliers.append(match)
        else:
            inliers.append(match)

    # Remove None entries from filtered keypoints lists
    keypoints1_filtered = [kp for kp in keypoints1_filtered if kp is not None]
    keypoints2_filtered = [kp for kp in keypoints2_filtered if kp is not None]

    return deviations, inliers, outliers, keypoints1_filtered, keypoints2_filtered


def init_matches(idx):
    """
        Initializes and matches keypoints between a pair of stereo images.
        Args:
        - idx (int): Index of the image pair.
        Returns:
        - img1_color (np.array): First image in color.
        - img2_color (np.array): Second image in color.
        - keypoints1 (list): List of keypoints in the first image.
        - keypoints2 (list): List of keypoints in the second image.
        - matches (list): List of matches between keypoints.
    """
    img1, img2 = read_images(idx)
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    keypoints1, descriptors1 = detect_keypoints(img1, method='ORB', num_keypoints=500)
    keypoints2, descriptors2 = detect_keypoints(img2, method='ORB', num_keypoints=500)
    matches = match_keypoints(descriptors1, descriptors2)
    return img1_color, img2_color, keypoints1, keypoints2, matches


def cv_triangulation(P0, P1, pts1, pts2):
    """
        Performs triangulation using OpenCV's triangulatePoints function and plots the 3D points.

        Args:
        - P0 (np.array): Projection matrix of the first camera (3x4).
        - P1 (np.array): Projection matrix of the second camera (3x4).
        - pts1 (np.array): Array of points in the first image.
        - pts2 (np.array): Array of points in the second image.

        Returns:
        - points_3D_cv (np.array): Array of triangulated 3D points.
    """
    points_3D_cv = cv2.triangulatePoints(P0, P1, pts1, pts2)
    points_3D_cv /= points_3D_cv[3]  # Normalize the points to make homogeneous coordinates 1
    points_3D_cv = points_3D_cv[:3].T  # Transpose to get an array of shape (N, 3)
    plot_3d_points(points_3D_cv, title="OpenCV Triangulation")
    return points_3D_cv


def cloud_points_triangulation(idx):
    """
    Performs triangulation of 3D points from stereo matches.

    Args:
    - idx (int): Index for image pair selection.

    Returns:
    - k (np.array): Intrinsic camera matrix.
    - P0 (np.array): Projection matrix for the first camera.
    - P1 (np.array): Projection matrix for the second camera.
    - points_3D_custom (np.array): Triangulated 3D points.
    """

    img1_color, img2_color, keypoints1, keypoints2, matches = init_matches(idx)
    deviations, inliers, _, kp_indices = reject_matches(keypoints1, keypoints2, matches)
    k, P0, P1 = (
        read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
    points_3D_custom, pts1, pts2 = triangulation_process(P0, P1, inliers, k, keypoints1, keypoints2)
    return k, P0, P1, points_3D_custom


def basic_match_with_significance_test():
    """
    Placeholder function for basic match with significance test.

    Currently not implemented.
    """

    pass


def create_dict_to_pnp_avish_test(matches_01, matches_11, filtered_keypoints_left1, filtered_keypoints_right1,
                                  points_3D_00):
    """
    Creates dictionaries for PnP based on filtered keypoints and matches.

    Args:
    - matches_01 (list): Matches between images 0 and 1.
    - matches_11 (list): Matches within image 1.
    - filtered_keypoints_left1 (list): Filtered keypoints in image 1.
    - filtered_keypoints_right1 (list): Filtered keypoints in image 1.
    - points_3D_00 (np.array): 3D points from image 0.

    Returns:
    - new_filtered_keypoints_left1_pts (np.array): 2D points of filtered keypoints in image 1.
    - new_filtered_keypoints_right1_pts (np.array): 2D points of corresponding keypoints in image 1.
    - new_filtered_3D_keypoints_left0 (np.array): Corresponding 3D points in image 0.
    """

    new_filtered_keypoints_left1 = []
    new_filtered_keypoints_right1 = []
    new_filtered_3D_keypoints_left0 = []
    i = 0
    dict_l1_to_r1 = {}
    for match in matches_11:
        dict_l1_to_r1[filtered_keypoints_left1[match.queryIdx]] = filtered_keypoints_right1[match.trainIdx]
    for match in matches_01:
        kp_left1 = filtered_keypoints_left1[match.trainIdx]
        new_filtered_keypoints_left1.append(kp_left1)
        kp_left0 = points_3D_00[match.queryIdx]
        new_filtered_3D_keypoints_left0.append(kp_left0)
        match.queryIdx = i
        match.trainIdx = i
        kp_right1 = dict_l1_to_r1[kp_left1]
        new_filtered_keypoints_right1.append(kp_right1)

    new_filtered_keypoints_left1_pts = [point.pt for point in new_filtered_keypoints_left1]
    new_filtered_keypoints_right1_pts = [point.pt for point in new_filtered_keypoints_right1]
    return (np.array(new_filtered_keypoints_left1_pts), np.array(new_filtered_keypoints_right1_pts),
            np.array(new_filtered_3D_keypoints_left0))


# add fitered_kp_right1, matches11.
def create_dict_to_pnp(matches_01, inliers_matches_11, filtered_keypoints_left1, keypoints_left0, keypoints_left1,
                       keypoints_right1,
                       points_3D_custom):
    """
    Creates dictionaries for PnP based on filtered keypoints and matches.

    Args:
    - matches_01 (list): Matches between images 0 and 1.
    - inliers_matches_11 (list): Inlier matches within image 1.
    - filtered_keypoints_left1 (list): Filtered keypoints in image 1.
    - keypoints_left0 (list): Keypoints in image 0.
    - keypoints_left1 (list): Keypoints in image 1.
    - keypoints_right1 (list): Keypoints in image 1.
    - points_3D_custom (np.array): 3D points from triangulation.

    Returns:
    - points_3d (np.array): 3D points.
    - points_2D_l0 (np.array): 2D points in image 0.
    - points_2D_l1 (np.array): 2D points in image 1.
    - points_2D_r1 (np.array): 2D points in image 1.
    """
    points_2Dleft1_to_2Dright1 = {}
    points_3d = []
    points_2D_l1 = []
    points_2D_r1 = []
    points_2D_l0 = []
    for match in inliers_matches_11:
        points_2Dleft1_to_2Dright1[keypoints_left1[match.queryIdx]] = keypoints_right1[match.trainIdx]
    for match in matches_01:
        # Get the index of the keypoint in the left1 image
        idx_2d_left1 = match.trainIdx
        kp_match_to_l1 = filtered_keypoints_left1[idx_2d_left1]
        pt_2d_r1 = points_2Dleft1_to_2Dright1[kp_match_to_l1].pt
        # get the index of the keypoint of left0 image
        # Get the index of the 3D point
        idx_3d = match.queryIdx
        pt_2d_l0 = keypoints_left0[idx_3d].pt
        # Get the 2D point from filtered_keypoints_left1
        pt_2d_l1 = filtered_keypoints_left1[idx_2d_left1].pt
        # Get the corresponding 3D point from points_3D_custom
        pt_3d = points_3D_custom[idx_3d]
        # Store the points in arrays
        points_3d.append(pt_3d)
        points_2D_l1.append(pt_2d_l1)
        points_2D_r1.append(pt_2d_r1)
        points_2D_l0.append(pt_2d_l0)
    return np.array(points_3d), np.array(points_2D_l0), np.array(points_2D_l1), np.array(points_2D_r1)


def create_in_out_l1_dict(inliers, points_2D_l1, filtered_keypoints_left1):

    in_out_l1_dict = {}
    for i, kp in enumerate(filtered_keypoints_left1):
        if kp.pt in points_2D_l1[inliers]:
            in_out_l1_dict[kp] = True
        else:
            in_out_l1_dict[kp] = False
    return in_out_l1_dict


def reject_matches_and_remove_keypoints1(keypoints1, keypoints2, matches):
    """
    Rejects matches based on vertical deviation between corresponding points.

    Args:
    - keypoints1 (list): List of keypoints in the first image.
    - keypoints2 (list): List of keypoints in the second image.
    - matches (list): List of matches between keypoints.

    Returns:
    - deviations (list): List of vertical deviations.
    - inliers (list): List of matches with deviations <= 2 pixels.
    - outliers (list): List of matches with deviations > 2 pixels.
    """
    deviations = []
    inliers = []
    outliers = []
    idx_inliers = []
    # Create copies of keypoints lists to avoid modifying the originals
    # Convert keypoints1 and keypoints2 to lists if they are tuples
    if isinstance(keypoints1, tuple):
        keypoints1 = list(keypoints1)
    if isinstance(keypoints2, tuple):
        keypoints2 = list(keypoints2)
    keypoints1_filtered = keypoints1.copy()
    keypoints2_filtered = keypoints2.copy()
    i = 0
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        deviation = abs(pt1[1] - pt2[1])  # Vertical deviation
        deviations.append(deviation)
        deviations.append(deviation)

        if deviation > 2:
            # Remove keypoints from filtered lists
            keypoints1_filtered[match.queryIdx] = None
            keypoints2_filtered[match.trainIdx] = None
            outliers.append(match)
        else:
            inliers.append(match)

    # Remove None entries from filtered keypoints lists
    keypoints1_filtered = [kp for kp in keypoints1_filtered if kp is not None]
    keypoints2_filtered = [kp for kp in keypoints2_filtered if kp is not None]

    return deviations, inliers, outliers, keypoints1_filtered, keypoints2_filtered


def stack_R_and_t(R, t):
    Rt = np.hstack((R, t))
    return Rt


def plot_camera_positions(extrinsic_matrices):
    # Define colors for each camera
    colors = ['r', 'g', 'b', 'c']

    # Extract camera positions from the extrinsic matrices
    positions = []
    for Rt in extrinsic_matrices:
        # The camera position is the negative inverse of the rotation matrix multiplied by the translation vector
        R = Rt[:3, :3]
        t = Rt[:3, 3]
        position = -np.linalg.inv(R).dot(t)
        positions.append(position)
        # print(position)

    positions = np.array(positions)

    # Plot the camera positions in 2D (x-z plane)
    plt.figure()
    for i, position in enumerate(positions):
        plt.scatter(position[0], position[2], c=colors[i], marker='o', label=f'Camera {i}')
        plt.text(position[0], position[2], f'Camera {i}', color=colors[i])

    # Set axis labels
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Camera Positions')

    # Set axis limits
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.grid(True)
    # plt.axis('equal')
    plt.legend()
    plt.show()


def read_cameras_matrices():
    with open(DATASET_PATH + '\\calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


def extract_keypoints_and_inliers(img_left, img_right):
    # Detect keypoints and compute descriptors
    keypoints_left, descriptors_left = DETECTOR.detectAndCompute(img_left, None)
    keypoints_right, descriptors_right = DETECTOR.detectAndCompute(img_right, None)

    # Match descriptors
    matches = MATCHER.match(descriptors_left, descriptors_right)

    # Filter matches based on the deviation threshold
    inliers = []
    outliers = []

    for match in matches:
        pt_left = keypoints_left[match.queryIdx]
        pt_right = keypoints_right[match.trainIdx]
        if abs(pt_left.pt[1] - pt_right.pt[1]) <= MAX_DEVIATION:
            inliers.append(match)
        else:
            outliers.append(match)
    inliers = sorted(inliers, key=lambda match: match.queryIdx)
    outliers = sorted(outliers, key=lambda match: match.queryIdx)
    return keypoints_left, descriptors_left, keypoints_right, descriptors_right, inliers, outliers


def cv_triangulate_matched_points(kps_left, kps_right, inliers,
                                  K, R_back_left, t_back_left, R_back_right, t_back_right):
    num_matches = len(inliers)
    pts1 = np.array([kps_left[inliers[i].queryIdx].pt for i in range(num_matches)])
    pts2 = np.array([kps_right[inliers[i].trainIdx].pt for i in range(num_matches)])

    proj_mat_left = K @ np.hstack((R_back_left, t_back_left))
    proj_mat_right = K @ np.hstack((R_back_right, t_back_right))

    X_4d = cv2.triangulatePoints(proj_mat_left, proj_mat_right, pts1.T, pts2.T)
    X_4d /= (X_4d[3] + 1e-10)

    return X_4d[:-1].T


def find_consensus_matches_indices(back_inliers, front_inliers, tracking_inliers):
    # Sort inliers based on their queryIdx, which is O(n log n) for each list
    back_sorted = sorted(back_inliers, key=lambda m: m.queryIdx)
    front_sorted = sorted(front_inliers, key=lambda m: m.queryIdx)

    # Create dictionaries to map queryIdx to their index in the sorted lists
    back_dict = {m.queryIdx: idx for idx, m in enumerate(back_sorted)}
    front_dict = {m.queryIdx: idx for idx, m in enumerate(front_sorted)}

    consensus = []
    # For each inlier in tracking_inliers, attempt to find the corresponding elements in back and front inliers
    for idx, inlier in enumerate(tracking_inliers):
        back_idx = inlier.queryIdx
        front_idx = inlier.trainIdx
        if back_idx in back_dict and front_idx in front_dict:
            idx_of_back = back_dict[back_idx]
            idx_of_front = front_dict[front_idx]
            consensus.append((idx_of_back, idx_of_front, idx))

    return consensus


def calculate_front_camera_matrix(cons_matches, back_points_cloud,
                                  front_inliers, front_kps_left, intrinsic_matrix):
    # Use cv2.solvePnP to compute the front-left camera's extrinsic matrix
    # based on at least 4 consensus matches and their corresponding 2D & 3D positions
    num_samples = len(cons_matches)
    if num_samples < 4:
        raise ValueError(f"Must provide at least 4 sampled consensus-matches, {num_samples} given")
    cloud_shape = back_points_cloud.shape
    assert cloud_shape[0] == 3 or cloud_shape[1] == 3, "Argument $back_points_cloud is not a 3D array"
    if cloud_shape[1] != 3:
        back_points_cloud = back_points_cloud.T  # making sure we have shape Nx3 for solvePnP
    points_3D = np.zeros((num_samples, 3))
    points_2D = np.zeros((num_samples, 2))

    # populate the arrays
    for i in range(num_samples):
        cons_match = cons_matches[i]
        points_3D[i] = back_points_cloud[cons_match[0]]
        front_left_matched_kp_idx = front_inliers[cons_match[1]].queryIdx
        points_2D[i] = front_kps_left[front_left_matched_kp_idx].pt

    success, rotation, translation = cv2.solvePnP(objectPoints=points_3D,
                                                  imagePoints=points_2D,
                                                  cameraMatrix=intrinsic_matrix,
                                                  distCoeffs=None,
                                                  flags=cv2.SOLVEPNP_EPNP)
    return success, cv2.Rodrigues(rotation)[0], translation


def calculate_right_camera_matrix(R_left, t_left, right_R0, right_t0):
    assert right_R0.shape == (3, 3) and R_left.shape == (3, 3)
    assert right_t0.shape == (3, 1) or right_t0.shape == (3,)
    assert t_left.shape == (3, 1) or t_left.shape == (3,)
    right_t0 = right_t0.reshape((3, 1))
    t_left = t_left.reshape((3, 1))

    front_right_Rot = right_R0 @ R_left
    front_right_trans = right_R0 @ t_left + right_t0
    assert front_right_Rot.shape == (3, 3) and front_right_trans.shape == (3, 1)
    return front_right_Rot, front_right_trans


def calculate_camera_locations(back_left_R, back_left_t, right_R0, right_t0,
                               cons_matches, back_points_cloud, front_inliers, front_kps_left, intrinsic_matrix):
    # Returns a 4x3 np array representing the 3D position of the 4 cameras,
    # in coordinates of the back_left camera (hence the first line should be np.zeros(3))
    back_right_R, back_right_t = calculate_right_camera_matrix(back_left_R, back_left_t, right_R0, right_t0)
    is_success = False
    while not is_success:
        cons_sample = random.sample(cons_matches, 4)
        is_success, front_left_R, front_left_t = calculate_front_camera_matrix(cons_sample, back_points_cloud,
                                                                               front_inliers, front_kps_left,
                                                                               intrinsic_matrix)
    front_right_R, front_right_t = calculate_right_camera_matrix(front_left_R, front_left_t, back_right_R, back_right_t)

    back_right_coordinates = - back_right_R.T @ back_right_t
    front_left_coordinates = - front_left_R.T @ front_left_t
    front_right_coordinates = - front_right_R.T @ front_right_t
    return np.array([np.zeros((3, 1)), back_right_coordinates,
                     front_left_coordinates, front_right_coordinates]).reshape((4, 3))


def calculate_pixels_for_3d_points(points_cloud_3d, intrinsic_matrix, Rs, ts):
    """
    Takes a collection of 3D points in the world and calculates their projection on the cameras' planes.
    The 3D points should be an array of shape 3xN.
    $Rs and $ts are rotation matrices and translation vectors and should both have length M.

    return: a Mx2xN np array of (p_x, p_y) pixel coordinates for each camera
    """
    assert len(Rs) == len(ts), \
        "Number of rotation matrices and translation vectors must be equal"
    assert points_cloud_3d.shape[0] == 3 or points_cloud_3d.shape[1] == 3, \
        f"Must provide a 3D points matrix, input has shape {points_cloud_3d.shape}"
    if points_cloud_3d.shape[0] != 3:
        points_cloud_3d = points_cloud_3d.T

    num_cameras = len(Rs)
    num_points = points_cloud_3d.shape[1]
    pixels = np.zeros((num_cameras, 2, num_points))
    for i in range(num_cameras):
        R, t = Rs[i], ts[i]
        t = np.reshape(t, (3, 1))
        projections = intrinsic_matrix @ (
                R @ points_cloud_3d + t)  # non normalized homogeneous coordinates of shape 3xN
        hom_coordinates = projections / (projections[2] + Epsilon)  # add epsilon to avoid 0 division
        pixels[i] = hom_coordinates[:2]
    return pixels


def extract_actual_consensus_pixels(cons_matches, back_inliers, front_inliers,
                                    back_left_kps, back_right_kps, front_left_kps, front_right_kps):
    # Returns a 4x2xN array containing the 2D pixels of all consensus-matched keypoints
    back_left_pixels, back_right_pixels = [], []
    front_left_pixels, front_right_pixels = [], []
    for m in cons_matches:
        # cons_matches is a list of tuples of indices: (back_inliers_idx, front_inlier_idx, tracking_match_idx)
        single_back_inlier, single_front_inlier = back_inliers[m[0]], front_inliers[m[1]]

        back_left_point = back_left_kps[single_back_inlier.queryIdx].pt
        back_left_pixels.append(np.array(back_left_point))

        back_right_point = back_right_kps[single_back_inlier.trainIdx].pt
        back_right_pixels.append(np.array(back_right_point))

        front_left_point = front_left_kps[single_front_inlier.queryIdx].pt
        front_left_pixels.append(np.array(front_left_point))

        front_right_point = front_right_kps[single_front_inlier.trainIdx].pt
        front_right_pixels.append(np.array(front_right_point))

    back_left_pixels = np.array(back_left_pixels).T
    back_right_pixels = np.array(back_right_pixels).T
    front_left_pixels = np.array(front_left_pixels).T
    front_right_pixels = np.array(front_right_pixels).T
    return np.array([back_left_pixels, back_right_pixels, front_left_pixels, front_right_pixels])


def find_supporter_indices_for_model(cons_3d_points, actual_pixels, intrinsic_matrix, Rs, ts, max_distance: int = 2):
    """
    Find supporters for the model ($Rs & $ts) our of all consensus-matches.
    A supporter is a consensus match that has a calculated projection (based on $Rs & $ts) that is "close enough"
    to it's actual keypoints' pixels in all four images. The value of "close enough" is the argument $max_distance

    Returns a list of consensus matches that support the current model.
    """

    # make sure we have a Nx3 cloud:
    cloud_shape = cons_3d_points.shape
    assert cloud_shape[0] == 3 or cloud_shape[1] == 3, "Argument $cons_3d_points is not a 3D-points array"
    if cloud_shape[1] != 3:
        cons_3d_points = cons_3d_points.T

    # calculate pixels for all four cameras and make sure it has correct shape
    calculated_pixels = calculate_pixels_for_3d_points(cons_3d_points.T, intrinsic_matrix, Rs, ts)
    assert actual_pixels.shape == calculated_pixels.shape

    # find indices that are no more than $max_distance apart on all 4 projections
    euclidean_distances = np.linalg.norm(actual_pixels - calculated_pixels, ord=2, axis=1)
    supporting_indices = np.where((euclidean_distances <= max_distance).all(axis=0))[0]
    return supporting_indices


def calculate_number_of_iteration_for_ransac(p: float, e: float, s: int) -> int:
    """
    Calculate how many iterations of RANSAC are required to get good enough results,
    i.e. for a set of size $s, with outlier probability $e and success probability $p
    we need N > log(1-$p) / log(1-(1-$e)^$s)

    :param p: float -> required success probability (0 < $p < 1)
    :param e: float -> probability to be outlier (0 < $e < 1)
    :param s: int -> minimal set size (s > 0)
    :return: N: int -> number of iterations
    """
    assert s > 0, "minimal set size must be a positive integer"
    nom = np.log(1 - p)
    denom = np.log(1 - np.power(1 - e, s))
    return int(nom / denom) + 1


def build_model(consensus_match_idxs, points_cloud_3d, front_inliers, kps_front_left,
                intrinsic_matrix, back_left_rot, back_left_trans, R0_right, t0_right, use_random=True):
    # calculate the model (R & t of each camera) based on
    # the back-left camera and the [R|t] transformation to Right camera
    back_right_rot, back_right_trans = calculate_right_camera_matrix(back_left_rot, back_left_trans, R0_right, t0_right)
    is_success = False
    while not is_success:
        sample_consensus_matches = random.sample(consensus_match_idxs, 4) if use_random else consensus_match_idxs
        is_success, front_left_rot, front_left_trans = calculate_front_camera_matrix(sample_consensus_matches,
                                                                                     points_cloud_3d, front_inliers,
                                                                                     kps_front_left, intrinsic_matrix)
    front_right_rot, front_right_trans = calculate_right_camera_matrix(front_left_rot, front_left_trans,
                                                                       R0_right, t0_right)
    Rs = [back_left_rot, back_right_rot, front_left_rot, front_right_rot]
    ts = [back_left_trans, back_right_trans, front_left_trans, front_right_trans]
    return Rs, ts


def estimate_projection_matrices_with_ransac(points_cloud_3d, cons_match_idxs,
                                             back_inliers, front_inliers,
                                             kps_back_left, kps_back_right,
                                             kps_front_left, kps_front_right,
                                             intrinsic_matrix,
                                             back_left_rot, back_left_trans,
                                             R0_right, t0_right,
                                             verbose: bool = False):
    """
    Implement RANSAC algorithm to estimate extrinsic matrix of the two front cameras,
    based on the two back cameras, the consensus-matches and the 3D points-cloud of the back pair.

    Returns the best fitting model:
        - Rs - rotation matrices of 4 cameras
        - ts - translation vectors of 4 cameras
        - supporters - subset of consensus-matches that support this model,
            i.e. projected keypoints are no more than 2 pixels away from the actual keypoint
    """
    start_time = time.time()
    success_prob = 0.99
    outlier_prob = 0.99  # this value is updated while running RANSAC
    num_iterations = calculate_number_of_iteration_for_ransac(0.99, outlier_prob, 4)

    prev_supporters_indices = []
    cons_3d_points = points_cloud_3d[[m[0] for m in cons_match_idxs]]
    actual_pixels = extract_actual_consensus_pixels(cons_match_idxs, back_inliers, front_inliers,
                                                    kps_back_left, kps_back_right, kps_front_left, kps_front_right)
    if verbose:
        print(f"Starting RANSAC with {num_iterations} iterations.")
        # todo: maybe change to i < ransac_bound or something else maybe like maor?
        #  see wat  in dept frames also worked
    while num_iterations > 0:
        Rs, ts = build_model(cons_match_idxs, points_cloud_3d, front_inliers, kps_front_left,
                             intrinsic_matrix, back_left_rot, back_left_trans, R0_right, t0_right, use_random=True)
        supporters_indices = find_supporter_indices_for_model(cons_3d_points, actual_pixels,
                                                              intrinsic_matrix, Rs, ts)

        if len(supporters_indices) > len(prev_supporters_indices):
            prev_supporters_indices = supporters_indices
            outlier_prob = 1 - len(prev_supporters_indices) / len(cons_match_idxs)
            num_iterations = calculate_number_of_iteration_for_ransac(0.99, outlier_prob, 4)
            if verbose:
                print(f"\tRemaining iterations: {num_iterations}\n\t\t" +
                      f"Number of Supporters: {len(prev_supporters_indices)}")
        else:
            num_iterations -= 1
            if verbose and num_iterations % 100 == 0:
                print(f"Remaining iterations: {num_iterations}\n\t\t" +
                      f"Number of Supporters: {len(prev_supporters_indices)}")

    # at this point we have a good model (Rs & ts) and we can refine it based on all supporters
    if verbose:
        print("Refining RANSAC results...")
    while True:
        curr_supporters = [cons_match_idxs[idx] for idx in prev_supporters_indices]
        Rs, ts = build_model(curr_supporters, points_cloud_3d, front_inliers, kps_front_left,
                             intrinsic_matrix, Rs[0], ts[0], R0_right, t0_right, use_random=False)
        supporters_indices = find_supporter_indices_for_model(cons_3d_points, actual_pixels, intrinsic_matrix, Rs, ts)
        if len(supporters_indices) > len(prev_supporters_indices):
            # we can refine the model even further
            prev_supporters_indices = supporters_indices
        else:
            # no more refinement, exit the loop
            break

    # finished, we can return the model
    curr_supporters = [cons_match_idxs[idx] for idx in prev_supporters_indices]
    elapsed = time.time() - start_time
    if verbose:
        print(f"RANSAC finished in {elapsed:.2f} seconds\n\tNumber of Supporters: {len(curr_supporters)}")

    elapsed = time.time() - start_time
    if verbose:
        print(f"RANSAC finished in {elapsed:.2f} seconds\n\tNumber of Supporters: {len(curr_supporters)}")

    return Rs, ts, curr_supporters


def transform_coordinates(points_3d, R, t):
    input_shape = points_3d.shape
    assert input_shape[0] == 3 or input_shape[1] == 3, \
        f"can only operate on matrices of shape 3xN or Nx3, provided {input_shape}"
    if input_shape[0] != 3:
        points_3d = points_3d.T  # making sure we are working with a 3xN array

    assert t.shape == (3, 1) or t.shape == (3,), \
        f"translation vector must be of size 3, provided {t.shape}"
    if t.shape != (3, 1):
        t = np.reshape(t, (3, 1))  # making sure we are using a 3x1 vector
    assert R.shape == (3, 3), f"rotation matrix must be of shape 3x3, provided {R.shape}"
    transformed = R @ points_3d + t
    assert transformed.shape == points_3d.shape
    return transformed


def estimate_complete_trajectory(num_frames: int = NUM_FRAMES, verbose=False):
    start_time, minutes_counter = time.time(), 0
    if verbose:
        print(f"Starting to process trajectory for {num_frames} tracking-pairs...")

    # load initiial cameras:
    K, M1, M2 = read_cameras_matrices()
    R0_left, t0_left = M1[:, :3], M1[:, 3:]
    R0_right, t0_right = M2[:, :3], M2[:, 3:]
    Rs_left, ts_left = [R0_left], [t0_left]

    # load first pair:
    img0_l, img0_r = read_images(0)
    back_pair_preprocess = extract_keypoints_and_inliers(img0_l, img0_r)
    back_left_kps, back_left_desc, back_right_kps, back_right_desc, back_inliers, _ = back_pair_preprocess

    for idx in range(1, num_frames):
        back_left_R, back_left_t = Rs_left[-1], ts_left[-1]
        back_right_R, back_right_t = calculate_right_camera_matrix(back_left_R, back_left_t, R0_right, t0_right)
        points_cloud_3d = cv_triangulate_matched_points(back_left_kps, back_right_kps, back_inliers,
                                                        K, back_left_R, back_left_t, back_right_R, back_right_t)

        # run the estimation on the current pair:
        front_left_img, front_right_img = read_images(idx)
        front_pair_preprocess = extract_keypoints_and_inliers(front_left_img, front_right_img)
        front_left_kps, front_left_desc, front_right_kps, front_right_desc, front_inliers, _ = front_pair_preprocess
        track_matches = sorted(MATCHER.match(back_left_desc, front_left_desc),
                               key=lambda match: match.queryIdx)
        consensus_indices = find_consensus_matches_indices(back_inliers, front_inliers, track_matches)
        curr_Rs, curr_ts, _ = estimate_projection_matrices_with_ransac(points_cloud_3d, consensus_indices, back_inliers,
                                                                       front_inliers, back_left_kps, back_right_kps,
                                                                       front_left_kps, front_right_kps, K,
                                                                       back_left_R, back_left_t, R0_right, t0_right,
                                                                       verbose=False)
        # print update if needed:
        curr_minute = int((time.time() - start_time) / 60)
        if verbose and curr_minute > minutes_counter:
            minutes_counter = curr_minute
            print(f"\tProcessed {idx} tracking-pairs in {minutes_counter} minutes")

        # update variables for the next pair:
        # todo: ask David if we need to bootstrap the kps
        Rs_left.append(curr_Rs[2])
        ts_left.append(curr_ts[2])
        back_left_kps, back_left_desc = front_left_kps, front_left_desc
        back_right_kps, back_right_desc = front_right_kps, front_right_desc
        back_inliers = front_inliers

    total_elapsed = time.time() - start_time
    if verbose:
        total_minutes = total_elapsed / 60
        print(f"Finished running for all tracking-pairs. Total runtime: {total_minutes:.2f} minutes")
    return Rs_left, ts_left, total_elapsed


def read_poses():
    Rs, ts = [], []
    file_path = os.path.join(os.getcwd(), r'dataset\poses\00.txt')
    f = open(file_path, 'r')
    for i, line in enumerate(f.readlines()):
        mat = np.array(line.split(), dtype=float).reshape((3, 4))
        Rs.append(mat[:, :3])
        ts.append(mat[:, 3:])
    return Rs, ts


def calculate_trajectory(Rs, ts):
    assert len(Rs) == len(ts), \
        "number of rotation matrices and translation vectors mismatch"
    num_samples = len(Rs)
    trajectory = np.zeros((num_samples, 3))
    for i in range(num_samples):
        R, t = Rs[i], ts[i]
        trajectory[i] -= (R.T @ t).reshape((3,))
    return trajectory


def compute_trajectory_and_distance(num_frames: int = NUM_FRAMES, verbose: bool = False):
    if verbose:
        print(f"\nCALCULATING TRAJECTORY FOR {num_frames} IMAGES\n")
    all_R, all_t, elapsed = estimate_complete_trajectory(num_frames, verbose=verbose)
    estimated_trajectory = calculate_trajectory(all_R, all_t)
    poses_R, poses_t = read_poses()
    ground_truth_trajectory = calculate_trajectory(poses_R[:num_frames], poses_t[:num_frames])
    distances = np.linalg.norm(estimated_trajectory - ground_truth_trajectory, ord=2, axis=1)
    return estimated_trajectory, ground_truth_trajectory, distances


def plot_inliers_outliers_ransac(consensus_match_indices_0_1, img0_left, img1_left, keypoints0_left, keypoints1_left,
                                 sup, tracking_matches):
    supporting_tracking_matches = [tracking_matches[idx] for (_a, _b, idx) in consensus_match_indices_0_1 if
                                   (_a, _b, idx) in sup]
    non_supporting_tracking_matches = [tracking_matches[idx] for (_a, _b, idx) in consensus_match_indices_0_1 if
                                       (_a, _b, idx) not in sup]

    supporting_pixels_back = [keypoints0_left[i].pt for i in [m.queryIdx for m in supporting_tracking_matches]]
    supporting_pixels_front = [keypoints1_left[i].pt for i in [m.trainIdx for m in supporting_tracking_matches]]
    non_supporting_pixels_back = [keypoints0_left[i].pt for i in [m.queryIdx for m in non_supporting_tracking_matches]]
    non_supporting_pixels_front = [keypoints1_left[i].pt for i in [m.trainIdx for m in non_supporting_tracking_matches]]

    # Start plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 20))  # Adjust the figsize as needed

    # Plot for img0_left
    axes[0].imshow(img0_left, cmap='gray')
    axes[0].scatter([x for (x, y) in supporting_pixels_back], [y for (x, y) in supporting_pixels_back], s=1, c='orange',
                    marker='*', label='Supporter')
    axes[0].scatter([x for (x, y) in non_supporting_pixels_back], [y for (x, y) in non_supporting_pixels_back], s=1,
                    c='cyan', marker='o', label='Non-Supporter')
    axes[0].axis('off')
    axes[0].set_title("Back Image (img0_left)")

    # Plot for img1_left
    axes[1].imshow(img1_left, cmap='gray')
    axes[1].scatter([x for (x, y) in supporting_pixels_front], [y for (x, y) in supporting_pixels_front], s=1,
                    c='orange', marker='*')
    axes[1].scatter([x for (x, y) in non_supporting_pixels_front], [y for (x, y) in non_supporting_pixels_front], s=1,
                    c='cyan', marker='o')
    axes[1].axis('off')
    axes[1].set_title("Front Image (img1_left)")

    # Add legend and title to the figure instead of the axes to avoid redundancy
    fig.legend(loc='lower center')
    fig.suptitle("Supporting & Non-Supporting Matches", fontsize=16)
    plt.tight_layout()  # Adjust the layout to make the plot compact
    plt.show()


def plot_two_3D_point_clouds(mR, mt, point_cloud_0):
    # create scatter plot of the two point clouds:
    point_cloud_0_transformed_to_1 = transform_coordinates(point_cloud_0.T, mR[2], mt[2])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(point_cloud_0.T[0], point_cloud_0.T[2],
                 point_cloud_0.T[1], c='b', s=2.5, marker='o', label='left0')
    ax.scatter3D(point_cloud_0_transformed_to_1[0], point_cloud_0_transformed_to_1[2],
                 point_cloud_0_transformed_to_1[1], c='r', s=2.5, marker='o', label='left1')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 20)
    ax.set_zlim(-4, 16)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    plt.legend()
    plt.show()


def read_images_from_dataset(idx: int):
    image_name = "{:06d}.png".format(idx)
    img0 = cv2.imread(DATASET_PATH + '\\image_0\\' + image_name, 0)
    img1 = cv2.imread(DATASET_PATH + '\\image_1\\' + image_name, 0)
    return img0, img1
import cv2
import numpy as np
from matplotlib import pyplot as plt

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

    img1 = cv2.imread(DATA_PATH + f'image_0/' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + f'image_1/' + img_name, 0)
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


def get_stereo_matches_with_filtered_keypoints_avish_test(img_left, img_right, feature_detector='AKAZE', max_deviation=2):
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
            #maybe we can do as follows:
            match.trainIdx = i
            match.queryIdx = i
            good_matches.append(match)
            i += 1
    filtered_descriptors_left = np.array(filtered_descriptors_left)
    filtered_descriptors_right = np.array(filtered_descriptors_right)

    return filtered_keypoints_left, filtered_keypoints_right, filtered_descriptors_left, filtered_descriptors_right, good_matches, keypoints_left, keypoints_right




def get_stereo_matches_with_filtered_keypoints(img_left, img_right, feature_detector='AKAZE', max_deviation=2):
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

    for match in matches:
        pt_left = keypoints_left[match.queryIdx].pt
        pt_right = keypoints_right[match.trainIdx].pt
        if abs(pt_left[1] - pt_right[1]) <= max_deviation:
            filtered_keypoints_left.append(keypoints_left[match.queryIdx])
            filtered_keypoints_right.append(keypoints_right[match.trainIdx])
            filtered_descriptors_left.append(descriptors_left[match.queryIdx])
            filtered_descriptors_right.append(descriptors_right[match.trainIdx])
            #maybe we can do as follows:
            #match.trainIdx = i (when i is the number of iteration)
            #match.queryIdx = i
            good_matches.append(match)

    filtered_descriptors_left = np.array(filtered_descriptors_left)
    filtered_descriptors_right = np.array(filtered_descriptors_right)

    return filtered_keypoints_left, filtered_keypoints_right, filtered_descriptors_left, filtered_descriptors_right, good_matches, keypoints_left, keypoints_right


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
    img1_color, img2_color, keypoints1, keypoints2, matches = init_matches(idx)
    deviations, inliers, _, kp_indices = reject_matches(keypoints1, keypoints2, matches)
    k, P0, P1 = (
        read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
    points_3D_custom, pts1, pts2 = triangulation_process(P0, P1, inliers, k, keypoints1, keypoints2)
    return k, P0, P1, points_3D_custom


def basic_match_with_significance_test():
    pass



def create_dict_to_pnp_avish_test(matches_01, matches_11, filtered_keypoints_left1, filtered_keypoints_right1, points_3D_00):
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

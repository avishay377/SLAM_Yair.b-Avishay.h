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


def triangulation_process(P0, P1, inliers, k, keypoints1, keypoints2):
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
    plot_3d_points(points_3D_custom, title="Custom Triangulation")
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


def plot_3d_points(points, title="3D Points"):
    """
        Plots 3D points using matplotlib.

        Args:
        - points (np.array): Array of 3D points.
        - title (str): Title of the plot (default is "3D Points").
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    plt.show()


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
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        deviation = abs(pt1[1] - pt2[1])  # Vertical deviation
        deviations.append(deviation)
        if deviation > 2:
            outliers.append(match)
        else:
            inliers.append(match)
    return deviations, inliers, outliers


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

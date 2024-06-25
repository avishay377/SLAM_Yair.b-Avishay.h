import cv2
import numpy as np
from algorithms_library import (
    cloud_points_triangulation,
    read_images,
    detect_keypoints, reject_matches,
    reject_matches_and_remove_keypoints, get_stereo_matches_with_filtered_keypoints, read_cameras,
    triangulation_process, create_dict_to_pnp
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_matches_without_rejection_from_imgs(first_img, second_img):
    keypoints_left, descriptors_left = detect_keypoints(first_img)
    keypoints_right, descriptors_right = detect_keypoints(second_img)

    bf = cv2.BFMatcher()
    matches = bf.match(descriptors_left, descriptors_left)
    return keypoints_left, keypoints_right, matches, descriptors_left


def get_matches_without_rejection(descriptors_left0, descriptors_left1):
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors_left0, descriptors_left1)
    return matches


def get_matches(first_img, second_img):
    keypoints_left, descriptors_left = detect_keypoints(first_img)
    keypoints_right, descriptors_right = detect_keypoints(second_img)

    bf = cv2.BFMatcher()
    matches = bf.match(descriptors_left, descriptors_right)
    _, inliers, _, keypoints_first_filtered, keypoints_second_filtered = reject_matches_and_remove_keypoints(
        keypoints_left, keypoints_right, matches)
    return keypoints_first_filtered, keypoints_second_filtered, inliers, descriptors_left


def find_common_keypoints(matches_01, matches_02, matches_03):
    list01 = []
    list02 = []
    list03 = []

    match_dict_01 = {m.queryIdx: m for m in matches_01}
    match_dict_02 = {m.trainIdx: m for m in matches_02}

    for m in matches_03:
        if m.queryIdx in match_dict_01 and m.trainIdx in match_dict_02:
            list01.append(match_dict_01[m.queryIdx])
            list02.append(match_dict_02[m.trainIdx])
            list03.append(m)

    return list01, list02, list03


def compute_extrinsic_matrix(points3D, points2D, K, flag=cv2.SOLVEPNP_AP3P):
    # Use cv2.solvePnP to compute the extrinsic camera matrix [R|t]
    # _, rvec, tvec = cv2.solvePnP
    # points2D = points2D.reshape(2, 1)
    # print("points3D:", points3D.shape, "points2D:", points2D.shape, "k:", K.shape)
    _, rvec, tvec = cv2.solvePnP(points3D, points2D, K, None, flags=flag)
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.hstack((R, tvec))
    t_left1 = tvec.flatten()
    return Rt, t_left1


def plot_camera_positions_0(extrinsic_matrices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Camera colors
    colors = ['r', 'g', 'b', 'c']

    # Plot camera positions
    for i, extrinsic_matrix in enumerate(extrinsic_matrices):
        #     # Extract R and t from extrinsic_matrix
        R = extrinsic_matrix[:, :3]
        t = extrinsic_matrix[:, 3]
        # Calculate camera position in global coordinates
        C = -np.dot(R.T, t)
        print(C)

        ax.scatter(C[0], C[1], C[2], color=colors[i], label=f'Camera {i + 1}')
        ax.text(C[0], C[1], C[2], f'Camera {i + 1}', color='black', fontsize=10, ha='center')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Relative Positions of Cameras')
    ax.legend()

    plt.show()


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
        print(position)

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
    plt.xlim(-1, 5)
    plt.ylim(-1, 2)

    plt.grid(True)
    # plt.axis('equal')
    plt.legend()
    plt.show()


def rectify(matches, key_points1, key_points2):
    idx_kp1 = {}
    # todo check of query is frame1
    matches_i_in_img1 = [m.queryIdx for m in matches]
    matches_i_in_img2 = [m.trainIdx for m in matches]
    for i, j in zip(matches_i_in_img1, matches_i_in_img2):
        if abs(key_points1[i].pt[1] - key_points2[j].pt[1]) < 2:
            idx_kp1[i] = j
    return idx_kp1


def q1():
    cloud_points_triangulation(0)
    cloud_points_triangulation(1)


#
# def q2_():
#     img_left0, img_right0 = read_images(0)
#     img_left1, img_right1 = read_images(1)
#
#     # Get matches for pair 0
#     keypoints_left0, keypoints_right0, matches_01, desc_00 = get_matches(img_left0, img_right0)
#     k, Rt_00, Rt_01, cloud_points_pair0, kp_indices = cloud_points_triangulation(0)
#     # Get matches for pair 1
#     keypoints_left1, keypoints_right1, matches_02, desc_10 = get_matches(img_left1, img_right1)
#
#     # Get matches between left0 and left1
#     keypoints_left1_1, descriptors_left1_1 = detect_keypoints(img_left1)
#     matches_03 = get_matches_without_rejection(desc_00, desc_10)
#
#     print("Matches in pair 0:", len(matches_01))
#     print("Matches in pair 1:", len(matches_02))
#     print("Matches between left0 and left1:", len(matches_03))
#
#     # Perform cloud triangulation for pair 0 (assuming this was already done in q1)
#     # k, Rt_00, Rt_01, cloud_points_pair0, kp_indices = cloud_points_triangulation(0)
#     k = np.array(k)
#
#     # matches_01 = [m for m in matches_01 if m.queryIdx in kp_indices.keys()]
#     # matches_03 = [m for m in matches_03 if m.queryIdx in kp_indices.keys()]
#     # matches_02 = [m for m in matches_02 if m.queryIdx in kp_indices.values()]
#     # Choose 4 key-points matched in all four images
#     points3D_pair0, points2D_left1 = find_kp_match_across_4_imgs(matches_01, matches_02, matches_03, cloud_points_pair0,
#                                                                  keypoints_left1_1)
#     print("Number of chosen key-points:", points3D_pair0.shape[0])
#
#     # keypoints_3d_pair0, keypoints_2d_img11 = (
#     #     method_name(cloud_points_pair0, k, keypoints_left1, matches_01, matches_02, matches_03))
#     # mutual_matches_ind_l0, mutual_matches_ind_l1 = get_mutual_kp_ind(ma)
#
#     Rt_10, t_10 = compute_extrinsic_matrix(points3D_pair0[:4], points2D_left1[:4], k)
#     print("Rt10: \n")
#     print(Rt_10)
#     # Compute Rt for right0 (already available as Rt_01)
#     R_01 = Rt_01[:, :3]
#     t_01 = Rt_01[:, 3]
#     R_11 = np.dot(Rt_10[:, :3], R_01)
#     t_11 = np.dot(Rt_10[:, :3], t_01) + t_10
#     Rt_11 = np.hstack((R_11, t_11.reshape(-1, 1)))
#     # location_camera
#     print(Rt_11)
#     # Compute R_right1 and t_right1
#     # R_right1 = np.dot(Rt_10[:, :3], R_01.T)
#     # t_right1 = t_10 - np.dot(R_right1, t_01)
#     # Rt_right_1 = np.hstack((R_right1, t_right1.reshape(-1, 1)))
#
#
#     # Plot camera positions
#     plot_camera_positions([Rt_00, Rt_10, Rt_01, Rt_11])
#     return Rt_10, t_10, points3D_pair0, points2D_left1

def method_name(cloud_points_pair0, k, keypoints_left1, matches_01, matches_02, matches_03):
    # Step 1: Find common key points across all four images
    # Create a dictionary to hold the match indices
    match_dict = {}
    # Populate the dictionary with matches from img00 to img01
    for m in matches_01:
        match_dict[m.queryIdx] = {'img01': m.trainIdx}
    # Update the dictionary with matches from img00 to img10
    for m in matches_03:
        if m.queryIdx in match_dict:
            match_dict[m.queryIdx]['img10'] = m.trainIdx
    # Update the dictionary with matches from img10 to img11
    for m in matches_02:
        for k, v in match_dict.items():
            if 'img10' in v and v['img10'] == m.queryIdx:
                match_dict[k]['img11'] = m.trainIdx
    # Filter out entries that don't have matches in all images
    common_matches = {k: v for k, v in match_dict.items() if len(v) == 3}
    # Select 4 key points (replace with your selection logic if needed)
    selected_indices = list(common_matches.keys())[:4]

    # Ensure the selected indices are within valid ranges
    selected_indices = [i for i in selected_indices if
                        i < len(cloud_points_pair0) and common_matches[i]['img11'] < len(keypoints_left1)]

    if len(selected_indices) < 4:
        raise ValueError("Not enough valid common key points found across all images.")

    # Extract 3D coordinates from cloud_points_pair0 for the selected key points
    keypoints_3d_pair0 = np.array([cloud_points_pair0[i] for i in selected_indices], dtype=np.float32)
    # Extract 2D coordinates in img11 for the selected key points
    keypoints_2d_img11 = np.array([keypoints_left1[common_matches[i]['img11']].pt for i in selected_indices],
                                  dtype=np.float32)

    print("Selected 3D keypoints from pair 0:\n", keypoints_3d_pair0)
    print("Corresponding 2D keypoints in img11:\n", keypoints_2d_img11)
    return keypoints_3d_pair0, keypoints_2d_img11


def get_mutual_kp_ind(matches00, matches11, matches01):
    mutual_kp_ind_l0 = []
    mutual_kp_ind_l1 = []
    ml0 = matches00.keys() & matches01.keys()
    for i in ml0:
        if matches01[i] in matches11.keys():
            mutual_kp_ind_l0.append(i)
            mutual_kp_ind_l1.append(matches01[i])
    return mutual_kp_ind_l0, mutual_kp_ind_l1


def find_kp_match_across_4_imgs(list01, list02, list03, cloud_points_pair0, keypoints_left1):
    """
    Choose 4 key-points that are matched on all four images.

    Args:
    - list01 (list): Matches between left0 and right0.
    - list02 (list): Matches between left1 and right1.
    - list03 (list): Matches between left0 and left1.
    - cloud_points_pair0 (numpy array): 3D points from pair 0 triangulation.

    Returns:
    - points3D_pair0 (numpy array): 3D coordinates of the chosen 4 key-points from pair 0.
    - points2D_left1 (list): List of 2D keypoints in left1 image corresponding to the chosen 4 key-points.
    """
    points3D_pair0 = []
    points2D_left1 = []

    match_dict_01 = {m.queryIdx: m for m in list01}
    match_dict_02 = {m.trainIdx: m for m in list02}

    for m in list03:
        if m.queryIdx in match_dict_01 and m.trainIdx in match_dict_02:
            # Get corresponding matches from list01 and list02
            match_left0_right0 = match_dict_01[m.queryIdx]
            match_left1_right1 = match_dict_02[m.trainIdx]

            # Choose this point
            point3D_pair0 = cloud_points_pair0[match_left0_right0.queryIdx]
            point2D_left1 = keypoints_left1[m.trainIdx].pt

            points3D_pair0.append(point3D_pair0)
            points2D_left1.append(point2D_left1)

    points2D_left1 = np.array(points2D_left1)
    print(points2D_left1.shape)
    points3D_pair0 = np.array(points3D_pair0)

    return points3D_pair0, points2D_left1


def project_points(points_3D, K, Rt):
    """
    Projects 3D points to 2D using camera matrix K and extrinsic matrix Rt

    Args:
    - points_3D: numpy array of shape (N, 3) containing 3D points (homogeneous)
    - K: intrinsic camera matrix (3x3)
    - Rt: extrinsic matrix (3x4)

    Returns:
    - points_2D: numpy array of shape (N, 2) containing 2D projected points
    """
    points_3D_hom = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))

    points_2D_hom = (points_3D_hom @ Rt.T @ K.T)
    points_2D = (points_2D_hom[:, :2] / points_2D_hom[:, [-1]])

    return points_2D
def plot_supporters(img_left0, img_left1, keypoints_left0, keypoints_left1, matches, supporters):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(cv2.cvtColor(img_left0, cv2.COLOR_BGR2RGB))
    axes[1].imshow(cv2.cvtColor(img_left1, cv2.COLOR_BGR2RGB))

    for i, match in enumerate(matches):
        pt_left0 = keypoints_left0[match.queryIdx].pt
        pt_left1 = keypoints_left1[match.trainIdx].pt

        color = 'cyan' if i in supporters else 'red'

        axes[0].plot(pt_left0[0], pt_left0[1], 'o', markersize=5, color=color)
        axes[1].plot(pt_left1[0], pt_left1[1], 'o', markersize=5, color=color)

    axes[0].set_title('Left Image 0')
    axes[1].set_title('Left Image 1')

    plt.show()



def plot_matches_with_supporters(img_left0, img_left1, keypoints_left0, keypoints_left1, matches, supporters_indices):
    img_left0_supporters = cv2.drawMatches(img_left0, keypoints_left0, img_left0, keypoints_left0, matches, None,
                                           matchesMask=supporters_indices, matchColor=(255, 0, 255),
                                           singlePointColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)
    img_left1_supporters = cv2.drawMatches(img_left1, keypoints_left1, img_left1, keypoints_left1, matches, None,
                                           matchesMask=supporters_indices, matchColor=(255, 0, 255),
                                           singlePointColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)

    # Convert BGR images to RGB for displaying with matplotlib
    img_left0_supporters_rgb = cv2.cvtColor(img_left0_supporters, cv2.COLOR_BGR2RGB)
    img_left1_supporters_rgb = cv2.cvtColor(img_left1_supporters, cv2.COLOR_BGR2RGB)

    # Plotting
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img_left0_supporters_rgb)
    plt.title('Left Image 0 with Matches and Supporters')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_left1_supporters_rgb)
    plt.title('Left Image 1 with Matches and Supporters')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def find_supporters(points_3D, keypoints_left0, keypoints_right0, keypoints_left1, keypoints_right1, K, Rt_00, Rt_10,
                    Rt_01,
                    Rt_11, threshold=2):
    """ Find supporters of the transformation T within a given distance threshold """
    supporters = []

    # Project points to all four images
    projected_left0 = project_points(points_3D, K, Rt_00)
    projected_right0 = project_points(points_3D, K, Rt_01)
    projected_left1 = project_points(points_3D, K, Rt_10)
    projected_right1 = project_points(points_3D, K, Rt_11)

    # Check distances for each point
    for i, (pt3D, pt_left0, pt_right0, pt_left1, pt_right1) in enumerate(
            zip(points_3D, keypoints_left0, keypoints_right0, keypoints_left1, keypoints_right1)):
        d_left0 = np.linalg.norm(projected_left0[i] - pt_left0.pt)
        d_right0 = np.linalg.norm(projected_right0[i] - pt_right0.pt)
        d_left1 = np.linalg.norm(projected_left1[i] - pt_left1.pt)
        d_right1 = np.linalg.norm(projected_right1[i] - pt_right1.pt)

        if d_left0 <= threshold and d_right0 <= threshold and d_left1 <= threshold and d_right1 <= threshold:
            supporters.append(i)

    return supporters


def q2():
    img_left0, img_right0 = read_images(0)
    img_left1, img_right1 = read_images(1)
    # Get matches for pair 0
    filtered_keypoints_left0, filtered_keypoints_right0, desc_00, _, matches_00, keypoints_left0, keypoints_right0 = (
        get_stereo_matches_with_filtered_keypoints(img_left0, img_right0))

    # Get matches for pair 1

    filtered_keypoints_left1, filtered_keypoints_right1, desc_10, _, matches_11, keypoints_left1, keypoints_right1 = (
        get_stereo_matches_with_filtered_keypoints(img_left1, img_right1))

    # Get matches between left0 and left1
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_01 = bf.match(desc_00, desc_10)

    # Perform cloud triangulation for pair 0 (assuming this was already done in q1)
    k, Rt_00, Rt_01 = (
        read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
    points_3D_custom, pts1, pts2 = (
        triangulation_process(Rt_00, Rt_01, matches_00, k, keypoints_left0, keypoints_right0))
    # Create the dictionary for PnP
    points_2D_to_3D_left1 = create_dict_to_pnp(matches_01, filtered_keypoints_left1, points_3D_custom)

    # Extract the first 4 points
    points_2D_list = list(points_2D_to_3D_left1.keys())
    points_3D_list = list(points_2D_to_3D_left1.values())

    # Convert the lists to the required format for compute_extrinsic_matrix
    points_2D_array = np.array(points_2D_list)
    points_3D_array = np.array(points_3D_list)
    print(points_3D_array, "\n", points_2D_array)
    # Use the first 4 points for extrinsic matrix computation
    Rt_10, t_10 = compute_extrinsic_matrix(points_3D_array[:4], points_2D_array[:4], k)

    # Compute Rt for right0 (already available as Rt_01)
    R_01 = Rt_01[:, :3]
    t_01 = Rt_01[:, 3]
    R_11 = np.dot(Rt_10[:, :3], R_01)
    t_11 = np.dot(Rt_10[:, :3], t_01) + t_10
    Rt_11 = np.hstack((R_11, t_11.reshape(-1, 1)))

    # Plot camera positions
    plot_camera_positions([Rt_00, Rt_10, Rt_01, Rt_11])

    # Find supporters of the transformation
    supporters = find_supporters(points_3D_array, keypoints_left0, keypoints_right0, keypoints_left1, keypoints_right1,
                                 k, Rt_00, Rt_10, Rt_01, Rt_11)

    plot_supporters(img_left0, img_left1, keypoints_left0, keypoints_left1, matches_01, supporters)





# Define the new function
def get_matched_points(matches_01, matches_02, matches_03, cloud_points_pair0, keypoints_left1_1):
    points3D_pair0 = []
    points2D_left1 = []

    for m in matches_03:
        if m.queryIdx < len(matches_01) and m.trainIdx < len(matches_02):
            idx_3d = matches_01[m.queryIdx].queryIdx
            idx_2d = matches_02[m.trainIdx].trainIdx

            if idx_3d < len(cloud_points_pair0) and idx_2d < len(keypoints_left1_1):
                points3D_pair0.append(cloud_points_pair0[idx_3d])
                points2D_left1.append(keypoints_left1_1[idx_2d].pt)

    return np.array(points3D_pair0), np.array(points2D_left1)


def q4(points3D_pair0, points2D_left1, Rt00):
    pass


def main():
    # q1()
    q2()
    # q2_()


if __name__ == '__main__':
    main()

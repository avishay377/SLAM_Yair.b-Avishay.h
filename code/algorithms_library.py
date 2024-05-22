import cv2


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

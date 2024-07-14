import cv2
import numpy as np
from tracking_database import TrackingDB
from algorithms_library import (compute_trajectory_and_distance_avish_test, plot_root_ground_truth_and_estimate,
                                read_images_from_dataset, read_cameras_matrices)
import random
import matplotlib.pyplot as plt

NUM_FRAMES = 3360  # or any number of frames you want to process


def plot_tracks(db: TrackingDB):
    all_tracks = db.all_tracks()
    track_lengths = [(trackId, len(db.frames(trackId))) for trackId in all_tracks if len(db.frames(trackId)) > 1]

    # 1. Longest track
    longest_track = max(track_lengths, key=lambda x: x[1])[0]

    # 2. Track with length of 10
    track_length_10 = next((trackId for trackId, length in track_lengths if length == 10), None)

    # 3. Track with length of 5
    track_length_5 = next((trackId for trackId, length in track_lengths if length == 5), None)

    # 4. Random track
    random_track = random.choice(track_lengths)[0]

    tracks_to_plot = [longest_track, track_length_10, track_length_5, random_track]
    track_names = ["longest_track", "track_length_10", "track_length_5", "random_track"]

    for trackId, track_name in zip(tracks_to_plot, track_names):
        if trackId is None:
            continue

        frames = db.frames(trackId)
        for frameId in frames:
            img_left, img_right = read_images_from_dataset(frameId)
            link = db.link(frameId, trackId)
            left_kp = link.left_keypoint()
            right_kp = link.right_keypoint()

            # Draw keypoints
            img_left = cv2.circle(img_left, (int(left_kp[0]), int(left_kp[1])), 5, (0, 255, 0), -1)  # Green color
            img_right = cv2.circle(img_right, (int(right_kp[0]), int(right_kp[1])), 5, (0, 255, 0), -1)  # Green color

            # Draw track lines
            if frameId > frames[0]:
                prev_frameId = frames[frames.index(frameId) - 1]
                prev_link = db.link(prev_frameId, trackId)
                prev_left_kp = prev_link.left_keypoint()
                prev_right_kp = prev_link.right_keypoint()

                img_left = cv2.line(img_left, (int(prev_left_kp[0]), int(prev_left_kp[1])),
                                    (int(left_kp[0]), int(left_kp[1])), (255, 0, 0), 2)  # Blue color
                img_right = cv2.line(img_right, (int(prev_right_kp[0]), int(prev_right_kp[1])),
                                     (int(right_kp[0]), int(right_kp[1])), (255, 0, 0), 2)  # Blue color

            # Save images
            cv2.imwrite(f"{track_name}frame{frameId}_left.png", img_left)
            cv2.imwrite(f"{track_name}frame{frameId}_right.png", img_right)


def calculate_statistics(db: TrackingDB):
    all_tracks = db.all_tracks()
    track_lengths = [len(db.frames(trackId)) for trackId in all_tracks if len(db.frames(trackId)) > 1]

    total_tracks = len(track_lengths)
    number_of_frames = db.frame_num()

    mean_track_length = np.mean(track_lengths) if track_lengths else 0
    max_track_length = np.max(track_lengths) if track_lengths else 0
    min_track_length = np.min(track_lengths) if track_lengths else 0

    frame_links = [len(db.tracks(frameId)) for frameId in db.all_frames()]
    mean_frame_links = np.mean(frame_links) if frame_links else 0

    return {
        "total_tracks": total_tracks,
        "number_of_frames": number_of_frames,
        "mean_track_length": mean_track_length,
        "max_track_length": max_track_length,
        "min_track_length": min_track_length,
        "mean_frame_links": mean_frame_links
    }


def q2(db):
    stats = calculate_statistics(db)
    print_statistics(stats)


def print_statistics(stats):
    print(f"Total number of tracks: {stats['total_tracks']}")
    print(f"Number of frames: {stats['number_of_frames']}")
    print(f"Mean track length: {stats['mean_track_length']}")
    print(f"Maximum track length: {stats['max_track_length']}")
    print(f"Minimum track length: {stats['min_track_length']}")
    print(f"Mean number of frame links: {stats['mean_frame_links']}")


def init_db():
    db = TrackingDB()
    estimated_trajectory, ground_truth_trajectory, distances, supporters_percentage = compute_trajectory_and_distance_avish_test(
        NUM_FRAMES, True, db)
    plot_root_ground_truth_and_estimate(estimated_trajectory, ground_truth_trajectory)
    # plot_tracks(db)
    db.serialize("db")
    # Load your database if necessary
    # db.load('db')
    return db, supporters_percentage


def q3(db):
    all_tracks = db.all_tracks()
    track_lengths = [(trackId, len(db.frames(trackId))) for trackId in all_tracks if len(db.frames(trackId)) >= 6]

    if not track_lengths:
        print("No tracks of length >= 6 found.")
        return

    track_to_display = track_lengths[2][0]
    frames = db.frames(track_to_display)

    fig, axes = plt.subplots(len(frames), 2, figsize=(10, len(frames) * 5))

    for idx, frameId in enumerate(frames):
        img_left, _ = read_images_from_dataset(frameId)
        link = db.link(frameId, track_to_display)
        left_kp = link.left_keypoint()

        x, y = int(left_kp[0]), int(left_kp[1])
        top_left_x = max(x - 10, 0)
        top_left_y = max(y - 10, 0)
        bottom_right_x = min(x + 10, img_left.shape[1])
        bottom_right_y = min(y + 10, img_left.shape[0])

        img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_GRAY2RGB)
        img_left_rgb = cv2.rectangle(img_left_rgb, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y),
                                     (0, 255, 0), 2)
        img_left_rgb = cv2.circle(img_left_rgb, (x, y), 2, (255, 0, 0), -1)  # Mark the feature

        # Crop the 20x20 region around the feature
        cropped_region = img_left_rgb[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        # Enlarge the cropped region
        enlarged_region = cv2.resize(cropped_region, (100, 100), interpolation=cv2.INTER_NEAREST)
        enlarged_region = cv2.circle(enlarged_region, (50, 50), 2, (255, 0, 0), -1)  # Mark the feature in zoomed view

        axes[idx, 0].imshow(img_left_rgb)
        axes[idx, 0].set_title(f"Frame {frameId}")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(enlarged_region)
        axes[idx, 1].set_title(f"Zoomed Feature in Frame {frameId}")
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()


def q4(db):
    all_frames = sorted(db.all_frames())
    connectivity = []

    for i in range(len(all_frames) - 1):
        current_frame = all_frames[i]
        next_frame = all_frames[i + 1]

        current_tracks = set(db.tracks(current_frame))
        next_tracks = set(db.tracks(next_frame))

        outgoing_tracks = current_tracks.intersection(next_tracks)
        connectivity.append(len(outgoing_tracks))

    # Plotting the connectivity graph
    plt.figure(figsize=(16, 8))  # Stretched x-axis by increasing the width
    plt.plot(range(len(connectivity)), connectivity, linestyle='-', color='blue', linewidth=0.5)  # Thinner blue line
    plt.xlabel('Frame Index')
    plt.ylabel('Number of Outgoing Tracks')
    plt.title('Connectivity Graph: Number of Outgoing Tracks per Frame')
    plt.grid(True)
    plt.xticks(
        np.arange(0, len(connectivity), step=max(1, len(connectivity) // 20)))  # Set x-ticks to be more spread out
    plt.tight_layout()
    plt.show()


def q5(supporters_percentage):
    # Create x-values starting from 1
    frames = list(range(1, len(supporters_percentage) + 1))

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(frames, supporters_percentage, marker='o', linestyle='-', color='b')
    plt.title('Percentage of Inliers per Frame')
    plt.xlabel('Frame')
    plt.ylabel('Percentage of Inliers')
    plt.xticks(frames)  # Ensure that each frame is labeled on the x-axis
    plt.grid(True)
    plt.show()


def q6(db):
    all_tracks = db.all_tracks()
    track_lengths = [len(db.frames(track_id)) for track_id in all_tracks if len(db.frames(track_id)) > 1]

    # Plotting the track length histogram
    plt.figure(figsize=(14, 8))
    plt.hist(track_lengths, bins=range(1, max(track_lengths) + 1), color='blue', edgecolor='black', alpha=0.7, log=True)
    plt.xlabel('Track Length')
    plt.ylabel('Frequency (log scale)')
    plt.title('Track Length Histogram')
    plt.grid(axis='y')
    plt.xticks(np.arange(1, max(track_lengths) + 1, step=5))
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
from tracking_database import TrackingDB
from algorithms_library import read_images, read_cameras, linear_least_square_pts

import numpy as np
import cv2
import random
import pickle



def parse_poses(poses):
    n = len(poses)
    transformations = []
    for i in range(n):
        T = np.eye(4)
        T[:3, :] = poses[i].reshape(3, 4)
        transformations.append(T)
    return transformations


def triangulate_point(link, selected_track_frames, poses):
    # Get the last frame in the track
    last_frame = selected_track_frames[-1]

    # Get the camera matrices
    T_last = poses[last_frame]

    # Extract the keypoints
    keypoint_left = link.left_keypoint()
    keypoint_right = link.right_keypoint()

    K, P_left, P_right = read_cameras_matrices()

    # Triangulate to get the 3D point in the last frame's coordinate system
    points_4D = cv2.triangulatePoints(P_left, P_right, keypoint_left, keypoint_right)
    point_3D = points_4D[:3] / points_4D[3]

    # Transform to world coordinates
    point_3D_world = T_last[:3, :3].T @ (point_3D - T_last[:3, 3])

    return point_3D_world


def compose_transformations(selected_track_frames, poses):
    T_composed = np.eye(4)
    for frame in selected_track_frames:
        T_composed = T_composed @ poses[frame]
    return T_composed


def project_point(point_3D_world, T, K):
    # Transform the point to the current frame's coordinate system
    point_3D_frame = T[:3, :3] @ point_3D_world + T[:3, 3]

    # Project the point
    point_2D = K @ point_3D_frame
    point_2D /= point_2D[2]

    return point_2D[:2]



if __name__ == "__main__":
    # db, supporters_percentage = init_db()
    # q2(db)
    # q3(db)
    # q4(db)
    # # print(supporters_percentage)
    # q6(db)
    # q5(supporters_percentage)

    # Load TrackingDB
    db = TrackingDB()
    db.load('db')

    # Load poses
    poses = np.loadtxt(r'C:\Users\avishay\PycharmProjects\SLAM_Yair.b-Avishay.h\code\dataset\poses\00.txt')

    # Parse the poses
    parsed_poses = parse_poses(poses)

    # Get all valid tracks
    valid_tracks = [track for track in db.all_tracks() if len(db.frames(track)) >= 10]

    # Select a random track of length >= 10
    selected_track = random.choice(valid_tracks)
    selected_track_frames = db.frames(selected_track)

    # Triangulate the 3D point using the last frame of the track
    link = db.link(selected_track_frames[-1], selected_track)
    point_3D
    point_3D_world = triangulate_point(link, selected_track_frames, parsed_poses)

    # Project to all frames in the track
    projections_left = {}
    projections_right = {}
    K, P_left, P_right = read_cameras_matrices()
    for frame in selected_track_frames:
        T_composed = compose_transformations(selected_track_frames[:frame + 1], parsed_poses)
        projections_left[frame] = project_point(point_3D_world, T_composed, K)

    # Reproject using the right camera projection matrix
    for frame in selected_track_frames:

        point_2D_left_homogeneous = np.append(projections_left[frame], 1)
        print(point_2D_left_homogeneous)
        print(P_right)
        point_2D_right = P_right @ point_2D_left_homogeneous
        point_2D_right /= point_2D_right[2]
        projections_right[frame] = point_2D_right[:2]

    # Calculate the reprojection error
    errors_left = []
    errors_right = []
    for frame in selected_track_frames:
        projected_point_left = projections_left[frame]
        projected_point_right = projections_right[frame]

        tracked_link = db.link(frame, selected_track)
        tracked_point_left = tracked_link.left_keypoint()
        tracked_point_right = tracked_link.right_keypoint()

        # Calculate reprojection errors
        error_left = np.linalg.norm(projected_point_left - tracked_point_left)
        error_right = np.linalg.norm(projected_point_right - tracked_point_right)

        errors_left.append(error_left)
        errors_right.append(error_right)

    mean_error_left = np.mean(errors_left)
    mean_error_right = np.mean(errors_right)
    print(f'Mean reprojection error (left): {mean_error_left}')
    print(f'Mean reprojection error (right): {mean_error_right}')

    pass
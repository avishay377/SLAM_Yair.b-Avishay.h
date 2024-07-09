import cv2
import numpy as np
from tracking_database import TrackingDB
from algorithms_library import (compute_trajectory_and_distance_avish_test, plot_root_ground_truth_and_estimate,
                                read_images_from_dataset)
import random
import matplotlib.pyplot as plt


NUM_FRAMES = 20  # or any number of frames you want to process


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
            cv2.imwrite(f"{track_name}_frame_{frameId}_left.png", img_left)
            cv2.imwrite(f"{track_name}_frame_{frameId}_right.png", img_right)
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
    # Load your database if necessary
    # db.load('your_database_filename')
    estimated_trajectory, ground_truth_trajectory, distances = compute_trajectory_and_distance_avish_test(3360, True, db)
    plot_root_ground_truth_and_estimate(estimated_trajectory, ground_truth_trajectory)
    plot_tracks(db)
    db.serialize("db")
    return db


def q3(db):
    pass


if __name__ == "__main__":
    db = init_db()
    q2(db)
    q3(db)
    
import glob
import json
import os
import pickle
import tkinter as tk
from tkinter import filedialog

import numpy as np

tablesize = (2.84, 1.42)
ball_radius = 0.0615 / 2


def extract_tracking_data_from_file(filename):
    """
    Extracts billiard shot tracking data from a single JSON file.

    For each shot (Entry) in every set, this function extracts the tracking data.
    It checks if tracking data exists, converts DeltaT_500us to seconds (500 µs = 0.0005 s),
    and converts the time, x, and y lists into NumPy arrays.

    The result for each shot is stored with:
        - "shotID": the shot ID.
        - "balls": a dict mapping each ball's index to its data arrays ("t", "x", and "y").
        - "filename": the name of the file from which the shot was extracted.

    Parameters:
        filename (str): The path to the JSON file.

    Returns:
        list: A list of dictionaries, one per shot.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    shots = []
    file_name = os.path.basename(filename)

    # Loop over each set in the match.
    for set_item in data.get("Match", {}).get("Sets", []):
        # Each shot is stored in the "Entries" list.
        for shot in set_item.get("Entries", []):
            # Check if tracking data exists.
            path_tracking = shot.get("PathTracking")
            if not path_tracking or not path_tracking.get("DataSets"):
                continue  # Skip shots without tracking data

            shot_id = shot.get("PathTrackingId")
            ball_data = {}

            # Process each ball's dataset.
            for dataset in path_tracking.get("DataSets", []):
                ball_index = dataset.get("BallColor")
                if not isinstance(ball_index, int):
                    print(f"Shot ID {shot_id}: Unrecognized ball index '{ball_index}'.")
                    continue  # Skip if the ball index is not recognized

                times, xs, ys = [], [], []
                coords = dataset.get("Coords")
                if not coords:
                    print(
                        f"Shot ID {shot_id}: No coordinates found for ball '{ball_index}'."
                    )
                    continue  # Skip if no coordinates are present

                for coord in coords:
                    times.append(coord.get("DeltaT_500us", 0) * 0.0005)
                    xs.append(coord.get("X") * tablesize[0])
                    ys.append(coord.get("Y") * tablesize[1])

                ball_data[ball_index] = {
                    "t": np.array(times),
                    "x": np.array(xs),
                    "y": np.array(ys),
                }

            if ball_data:
                shots.append(
                    {"shotID": shot_id, "balls": ball_data, "filename": file_name}
                )
            else:
                print(f"Shot ID {shot_id}: No valid ball data found.")

    return shots


def extract_all_shots_from_folder(folder):
    all_shots = []
    json_files = glob.glob(os.path.join(folder, "*.json"))

    for json_file in json_files:
        shots = extract_tracking_data_from_file(json_file)
        print(f"File: {os.path.basename(json_file)} - Shots extracted: {len(shots)}")
        all_shots.extend(shots)

    return all_shots


def extract_all_shots_from_files(files):
    all_shots = []
    for file in files:
        shots = extract_tracking_data_from_file(file)
        print(f"File: {os.path.basename(file)} - Shots extracted: {len(shots)}")
        all_shots.extend(shots)
    return all_shots


# function to check for all shots whether in that shot at least 1 ball has more than 10 data points
# if for that shot one ball has more than 10 data points, then the shot is valid
# if for that shot no ball has more than 10 data points, then the shot is invalid
# also check if all ball have at least 1 data point
# if a ball has no data points, then the shot is invalid
# delete invalid shots
# optimize for speed by using numpy indexing
def check_datalength(all_shots):
    print("Data length check started ...")
    valid_shots = []
    for shot in all_shots:
        valid = True
        for ball in shot["balls"].values():
            if len(ball["t"]) < 1:
                valid = False
                break
        if valid:
            valid_shots.append(shot)
    return valid_shots


# function to delete all data points outside of the table
# the table has a size of 2.84m x 1.42m
# if a data point is outside of the table, it will be deleted
# consider the ball radius of 0.0615/2 = 0.03075m
# use for fast exceution indexing
# return the corrected all_shots
def delete_outside_table(all_shots):
    print("Outside points deletion started ...")
    for shot in all_shots:
        for ball in shot["balls"].values():
            index = np.where(
                (ball["x"] < ball_radius)
                | (ball["x"] > tablesize[0] - ball_radius)
                | (ball["y"] < ball_radius)
                | (ball["y"] > tablesize[1] - ball_radius)
            )
            ball["x"] = np.delete(ball["x"], index)
            ball["y"] = np.delete(ball["y"], index)
            ball["t"] = np.delete(ball["t"], index)
    return all_shots


# project datapoints outside of playground on to the cushion
# if a ball is outside of the playground, it will be projected onto the cushion
# use linear interpolation to find the intersection point with the cushion, starting from the last point inside the playground
# consider the ball radius
# optimize for speed by using numpy indexing
# return the corrected all_shots
def project_data_outside_table(all_shots):
    print("Outside points projection started ...")
    for shot in all_shots:
        for ball_index, ball in shot["balls"].items():
            index = np.where(
                (ball["x"] < ball_radius)
                | (ball["x"] > tablesize[0] - ball_radius)
                | (ball["y"] < ball_radius)
                | (ball["y"] > tablesize[1] - ball_radius)
            )
            if len(index[0]) > 0:
                # correct the time also for the projection
                for i in index[0]:
                    x = ball["x"][i]
                    y = ball["y"][i]
                    t = ball["t"][i]
                    x_old = ball["x"][i - 1]
                    y_old = ball["y"][i - 1]
                    t_old = ball["t"][i - 1]
                    x_new = x
                    y_new = y
                    tnew = t

                    if x < ball_radius:
                        x_new = ball_radius
                        y_new = y
                        # y_new = y_old + (y-y_old)/(x-x_old)*(x_new-x_old)
                        # tnew = t_old + (t-t_old)/(x-x_old)*(x_new-x-old)
                    if x > tablesize[0] - ball_radius:
                        x_new = tablesize[0] - ball_radius
                        y_new = y
                        # y_new = y_old + (y-y_old)/(x-x-old)*(x_new-x-old)
                        # tnew = t_old + (t-t-old)/(x-x-old)*(x_new-x-old)
                    if y < ball_radius:
                        y_new = ball_radius
                        x_new = x
                        # x_new = x_old + (x-x-old)/(y-y-old)*(y_new-y-old)
                        # tnew = t_old + (t-t-old)/(y-y-old)*(y_new-y-old)
                    if y > tablesize[1] - ball_radius:
                        y_new = tablesize[1] - ball_radius
                        x_new = x
                        # x_new = x_old + (x-x-old)/(y-y-old)*(y_new-y-old)
                        # tnew = t_old + (t-t-old)/(y-y-old)*(y_new-y-old)

                    ball["x"][i] = x_new
                    ball["y"][i] = y_new

    return all_shots


# function to correct overlapping balls only in initial position
# if balls are overlapping in the initial position, they will be moved apart
# the balls will be moved in direction of the centerpoints of the balls
# the distance moved is half of the overlapping distance for each ball
# optimize for speed by using numpy indexing
# return the corrected all_shots


def correct_overlapping_balls(all_shots):
    print("Overlapping balls correction started ...")
    shots_to_delete = []
    for shot in all_shots:
        balls = list(shot["balls"].values())
        for i in range(1, len(balls)):
            for j in range(i):
                ball1 = balls[i]
                ball2 = balls[j]
                x1 = ball1["x"][0]
                y1 = ball1["y"][0]
                x2 = ball2["x"][0]
                y2 = ball2["y"][0]

                # Calculate the distance between the centers of the balls
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) - 2 * ball_radius

                if distance < 0:
                    # Balls are interfering
                    # Move them in the direction of the ball center connection
                    vec = np.array([x2 - x1, y2 - y1])
                    centerdistance = np.linalg.norm(vec)
                    if centerdistance == 0:
                        print(
                            f"Shot ID {shot['shotID']}: Balls are at the same position. Deleting shot."
                        )
                        shots_to_delete.append(shot)
                        break

                    vec = -vec / centerdistance * distance

                    ball1["x"][0] = x1 - vec[0] / 2
                    ball1["y"][0] = y1 - vec[1] / 2
                    ball2["x"][0] = x2 + vec[0] / 2
                    ball2["y"][0] = y2 + vec[1] / 2

            if shot in shots_to_delete:
                break

    # Remove shots that need to be deleted
    all_shots = [shot for shot in all_shots if shot not in shots_to_delete]
    return all_shots


# function to check and correct monotinc increase of time
# if the time is not increasing, it will be corrected by deleteing the data points
# that are not increasing
# optimize for speed by using numpy indexing
# return the corrected all_shots
def correct_time_flow(all_shots):
    print("Time flow correction started ...")
    corrected_shots = []
    for shot in all_shots:
        shot_corrected = True
        for ball in shot["balls"].values():
            t = ball["t"]
            x = ball["x"]
            y = ball["y"]

            dt = np.diff(t)
            ind = np.where(dt <= 0)[0]
            delind = []

            # Try to correct
            if len(ind) > 0:
                for ei in ind:
                    delindnew = np.where(t[ei + 1 :] <= t[ei])[0] + ei + 1
                    delind.extend(delindnew)

                t = np.delete(t, delind)
                x = np.delete(x, delind)
                y = np.delete(y, delind)

                dt = np.diff(t)
                indnew = np.where(dt <= 0)[0]

                if len(indnew) == 0:
                    ball["t"] = t
                    ball["x"] = x
                    ball["y"] = y
                else:
                    shot_corrected = False
                    break

        if shot_corrected:
            corrected_shots.append(shot)

    return corrected_shots


# check time to start at 0
# if the time does not start at 0, shot shot be deleted
# optimize for speed by using numpy indexing
# return the corrected all_shots
def check_time_start(all_shots):
    print("Time start check started ...")
    valid_shots = []
    for shot in all_shots:
        valid = True
        for ball in shot["balls"].values():
            if ball["t"][0] != 0:
                valid = False
                break
        if valid:
            valid_shots.append(shot)

    return valid_shots


# function to check whether end time is the same for all balls
# if the end time is not the same for all balls, the shot is invalid
# the shot will be deleted
# optimize for speed by using numpy indexing
# return the
def check_same_endtime(all_shots):
    print("End time check started ...")
    valid_shots = []
    for shot in all_shots:
        valid = True
        endtime = None
        for ball in shot["balls"].values():
            if endtime is None:
                endtime = ball["t"][-1]
            else:
                if ball["t"][-1] != endtime:
                    valid = False
                    break
        if valid:
            valid_shots.append(shot)

    return valid_shots


# create a function to correct wrong data points.
# Some single points can have jumps due to noise.
# these datapoints must be deleted.
def correct_wrong_data_points(all_shots):
    print("Wrong data points correction started ...")
    for shot in all_shots:
        for ball in shot["balls"].values():
            t = ball["t"]
            x = ball["x"]
            y = ball["y"]

            i = 0
            while i < len(t) - 2:
                dsx_02 = x[i + 2] - x[i]
                dsy_02 = y[i + 2] - y[i]
                dl0 = np.sqrt(dsx_02**2 + dsy_02**2)

                dsx_01 = x[i + 1] - x[i]
                dsy_01 = y[i + 1] - y[i]
                dl1 = np.sqrt(dsx_01**2 + dsy_01**2)

                dsx_12 = x[i + 2] - x[i + 1]
                dsy_12 = y[i + 2] - y[i + 1]
                dl2 = np.sqrt(dsx_12**2 + dsy_12**2)

                if dl0 < dl1 * 0.5 and dl0 < dl2 * 0.5 and dl1 > 0.1:
                    t = np.delete(t, i + 1)
                    x = np.delete(x, i + 1)
                    y = np.delete(y, i + 1)
                else:
                    i += 1

            ball["t"] = t
            ball["x"] = x
            ball["y"] = y

    return all_shots


# function to check time gaps
def check_gaps_in_tracking(all_shots, no_data_distance_limit, max_velocity):
    print("Gaps in tracking check started ...")
    valid_shots = []
    for shot in all_shots:
        shot_valid = True
        for ball in shot["balls"].values():
            t = ball["t"]
            x = ball["x"]
            y = ball["y"]

            ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
            vabs = ds / np.diff(t)

            if np.any(ds > no_data_distance_limit):
                print(f"Shot ID {shot['shotID']}: Gap in data is too big.")
                shot_valid = False
                break

            if np.any(vabs > max_velocity):
                print(f"Shot ID {shot['shotID']}: Velocity is too high.")
                shot_valid = False
                break

            if shot["shotID"] == 392699:
                # create a plot with the velocity
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.plot(t[:-1], ds)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("ds (m)")
                ax.set_title(f"Shot ID {shot['shotID']}")
                plt.show()

            if shot["shotID"] == 392699:
                # create a plot with the velocity
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.plot(t[:-1], vabs)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Velocity (m/s)")
                ax.set_title(f"Shot ID {shot['shotID']}")
                plt.show()

        if shot_valid:
            valid_shots.append(shot)

    return valid_shots


# function to do all checks and correction
def check_and_correct(all_shots):
    all_shots = check_datalength(all_shots)
    print(f"\nTotal remaining after checks: {len(all_shots)}")
    # all_shots = delete_outside_table(all_shots)
    all_shots = project_data_outside_table(all_shots)
    print(f"\nTotal remaining after checks: {len(all_shots)}")
    all_shots = correct_overlapping_balls(all_shots)
    print(f"\nTotal remaining after checks: {len(all_shots)}")
    all_shots = correct_time_flow(all_shots)
    print(f"\nTotal remaining after checks: {len(all_shots)}")
    all_shots = check_time_start(all_shots)
    print(f"\nTotal remaining after checks: {len(all_shots)}")
    all_shots = check_same_endtime(all_shots)
    print(f"\nTotal remaining after checks: {len(all_shots)}")
    all_shots = correct_wrong_data_points(all_shots)
    print(f"\nTotal remaining after checks: {len(all_shots)}")
    all_shots = check_gaps_in_tracking(all_shots, 0.4, 10)
    print(f"\nTotal remaining after checks: {len(all_shots)}")
    return all_shots


def angle_between_vectors(v1, v2):
    """Calculate the angle between two vectors."""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi  # Convert to degrees


def extract_b1_b2_b3(all_shots):
    for shot in all_shots:
        balls = shot["balls"]
        ball_colors = list(balls.keys())

        if len(ball_colors) < 3:
            print(f"Shot ID {shot['shotID']}: Not enough balls to identify b1, b2, b3.")
            continue

        # Check if any ball has less than 2 data points
        if any(len(balls[color]["t"]) < 2 for color in ball_colors):
            print(
                f"Shot ID {shot['shotID']}: Not enough data points to identify b1, b2, b3."
            )
            continue

        # Sort balls by the time of their second data point
        sorted_colors = sorted(ball_colors, key=lambda color: balls[color]["t"][1])

        # Temporary assumed B1B2B3 order
        b1_color = sorted_colors[0]
        b2_color = sorted_colors[1]
        b3_color = sorted_colors[2]

        b1 = balls[b1_color]
        b2 = balls[b2_color]
        b3 = balls[b3_color]

        if len(b1["t"]) >= 3:
            tb2i2 = np.where(b2["t"][1:] <= b1["t"][1])[0]
            tb3i2 = np.where(b3["t"][1:] <= b1["t"][1])[0]
            tb2i3 = np.where(b2["t"][1:] <= b1["t"][2])[0]
            tb3i3 = np.where(b3["t"][1:] <= b1["t"][2])[0]

            if (
                len(tb2i2) == 0
                and len(tb3i2) == 0
                and len(tb2i3) == 0
                and len(tb3i3) == 0
            ):
                # Only B1 moved in the first time step for sure
                shot["balls"] = {b1_color: b1, b2_color: b2, b3_color: b3}
                continue

            if (len(tb2i2) > 0 or len(tb2i3) > 0) and (
                len(tb3i2) == 0 and len(tb3i3) == 0
            ):
                # B1 and B2 moved
                vec_b1b2 = np.array([b2["x"][0] - b1["x"][0], b2["y"][0] - b1["y"][0]])
                vec_b1dir = np.array([b1["x"][1] - b1["x"][0], b1["y"][1] - b1["y"][0]])
                vec_b2dir = np.array([b2["x"][1] - b2["x"][0], b2["y"][1] - b2["y"][0]])

                angle_b1 = angle_between_vectors(vec_b1b2, vec_b1dir)
                angle_b2 = angle_between_vectors(vec_b1b2, vec_b2dir)

                if angle_b2 > 90:
                    b1_color, b2_color = b2_color, b1_color
                    b1, b2 = b2, b1

            if (len(tb2i2) == 0 and len(tb2i3) == 0) and (
                len(tb3i2) > 0 or len(tb3i3) > 0
            ):
                # B1 and B3 moved
                vec_b1b3 = np.array([b3["x"][0] - b1["x"][0], b3["y"][0] - b1["y"][0]])
                vec_b1dir = np.array([b1["x"][1] - b1["x"][0], b1["y"][1] - b1["y"][0]])
                vec_b3dir = np.array([b3["x"][1] - b3["x"][0], b3["y"][1] - b3["y"][0]])

                angle_b1 = angle_between_vectors(vec_b1b3, vec_b1dir)
                angle_b3 = angle_between_vectors(vec_b1b3, vec_b3dir)

                if angle_b3 > 90:
                    b1_color, b3_color = b3_color, b1_color
                    b1, b3 = b3, b1

        shot["balls"] = {
            b1_color: balls[b1_color],
            b2_color: balls[b2_color],
            b3_color: balls[b3_color],
        }

    return all_shots


def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(
        title="Select JSON Files", filetypes=[("JSON files", "*.json")]
    )
    return file_paths


if __name__ == "__main__":
    file_paths = select_files()

    all_shots = []

    if file_paths:
        all_shots.extend(extract_all_shots_from_files(file_paths))

    print(f"\nTotal shots extracted from all files and folders: {len(all_shots)}")

    # Sort shots by shotID
    all_shots.sort(key=lambda x: x["shotID"])

    # Save to a file
    with open("all_shots_raw.pkl", "wb") as f:
        pickle.dump(all_shots, f)

    # Load the shots from the file
    with open("all_shots_raw.pkl", "rb") as f:
        shots = pickle.load(f)

    print(f"\nTotal shots loaded from file: {len(shots)}")

    # Check and correct the shots
    shots = check_and_correct(shots)

    # Save to a file
    with open("all_shots.pkl", "wb") as f:
        pickle.dump(shots, f)

    # Identify b1, b2, b3
    print("\nB1, B2, B3 identification started ...")
    shots = extract_b1_b2_b3(shots)

    print(f"\nTotal remaining after checks: {len(shots)}")

    # Save to a file
    with open("all_shots.pkl", "wb") as f:
        pickle.dump(shots, f)

    # # Load the shots from the file
    # with open("all_shots.pkl", "rb") as f:
    #     shots = pickle.load(f)

    # # Example: Access the first shot
    # print("Shot ID:", shots[0]["shotID"])
    # print("From File:", shots[0]["filename"])

    # # Example: Access ball tracking data
    # for ball_color, data in shots[0]["balls"].items():
    #     print(f"Ball {ball_color}:")
    #     print("  Time (s):", data["t"])
    #     print("  X (m):", data["x"])
    #     print("  Y (m):", data["y"])

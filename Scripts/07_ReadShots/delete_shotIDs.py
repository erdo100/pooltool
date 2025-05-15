# filepath: /Users/ersindogan/3cushiontool/Scripts/clean_shots.py
import os
import pickle

FILE_NAME = "all_shots.pkl"
MAX_SHOT_ID = 394188

print("Current working directory:", os.getcwd())


def main():
    try:
        with open(FILE_NAME, "rb") as f:
            all_shots = pickle.load(f)
    except FileNotFoundError:
        print(f"Could not find file: {FILE_NAME}")
        return

    original_count = len(all_shots)
    filtered_shots = [
        shot for shot in all_shots if shot.get("shotID", 0) <= MAX_SHOT_ID
    ]
    removed_count = original_count - len(filtered_shots)

    with open(FILE_NAME, "wb") as f:
        pickle.dump(filtered_shots, f)

    print(f"Loaded {original_count} shots from {FILE_NAME}.")
    print(f"Removed {removed_count} shots with shotID > {MAX_SHOT_ID}.")
    print(f"{len(filtered_shots)} shots remain in {FILE_NAME}.")


if __name__ == "__main__":
    main()

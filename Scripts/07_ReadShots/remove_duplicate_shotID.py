import os
import pickle
import tkinter as tk
from tkinter import filedialog


def load_and_clean_shots():
    # Open file picker to select a pickle file
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    if not file_path:
        return

    # Load the pickle file
    with open(file_path, "rb") as f:
        all_shots = pickle.load(f)

    # Scan for duplicate shotIDs and remove duplicates
    seen_shot_ids = set()
    unique_shots = []
    for shot in all_shots:
        shot_id = shot["shotID"]
        if shot_id not in seen_shot_ids:
            unique_shots.append(shot)
            seen_shot_ids.add(shot_id)
        else:
            print(f"Duplicate shotID found: {shot_id}")

    # Create a new filename with _1 added
    base, ext = os.path.splitext(file_path)
    new_file_path = base + "_1" + ext

    # Save the cleaned data to the new file
    with open(new_file_path, "wb") as f:
        pickle.dump(unique_shots, f)

    print(f"Cleaned data saved to {new_file_path}")


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    load_and_clean_shots()

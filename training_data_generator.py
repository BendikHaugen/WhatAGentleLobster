from pathlib import Path
import pickle
from itertools import permutations
from typing import List, Tuple
from collections import Counter

from config import Config
from lobster import Lobster


def lobsters_to_samples(
    window_size: int,
    lobster_file_path: str,
):
    lobsters = Lobster.from_json(lobster_file_path)

    num_frames = len(lobsters[0].boundig_boxes)
    num_lobsters = len(lobsters)

    x_data = []
    y_data = []
    for i in range(window_size - 1, num_frames):
        for perm in permutations(lobsters, 2):
            perm: Tuple[Lobster, Lobster]
            l1, l2 = perm

            # if (
            #     utilities.distance(l1.centers[-1], l2.centers[-1])
            #     >= Config.Main.minimum_distance
            # ):
            #     continue

            # x = Lobster.get_x_data(l1, l2, i, window_size)
            x = Lobster.get_x_data_kp(l1, l2, i, window_size)
            y = Lobster.get_y_data_l1_attacks_l2(l1, l2, i)
            x_data.append(x)
            y_data.append(y)

    print(f"Fil {lobster_file_path}")
    print(f"Antall hummer:     {num_lobsters :3d}")
    print(f"Antall frames:     {num_frames :3d}")
    print(f"Antall features:    {len(x_data[0])}")
    print(f"Antall samples:  {len(x_data)}")
    print()

    # print("Første sample:")
    # print(x_data[0])
    # print(y_data[0])
    print("Y-data counter:")
    print(Counter(y_data))
    print()
    return x_data, y_data


def main():
    output_name = Config.TrainingDataGenerator.output_file_name
    window_size = Config.TrainingDataGenerator.window_size

    annotations_base_path = Config.TrainingDataGenerator.lobster_base_path
    annotations_file_ending = ".json"
    output_base_path = Config.TrainingDataGenerator.output_base_path
    output_file_ending = ".pkl"

    lob_file_names_train: List[str] = Config.TrainingDataGenerator.lobs_file_names_train
    lob_file_names_val: List[str] = Config.TrainingDataGenerator.lobs_file_names_val

    if not Path(output_base_path).exists():
        Path(output_base_path).mkdir(parents=True)

    for suffix, file_name_list in [
        ("_train", lob_file_names_train),
        ("_val", lob_file_names_val),
    ]:
        x_data = []
        y_data = []
        for lobster_name in file_name_list:
            lobster_file_path = (
                annotations_base_path + "/" + lobster_name + annotations_file_ending
            )
            x, y = lobsters_to_samples(window_size, lobster_file_path)
            x_data += x
            y_data += y

        with open(
            output_base_path + "/" + output_name + "_x" + suffix + output_file_ending,
            "wb",
        ) as x_fil:
            pickle.dump(x_data, x_fil, -1)
        with open(
            output_base_path + "/" + output_name + "_y" + suffix + output_file_ending,
            "wb",
        ) as y_fil:
            pickle.dump(y_data, y_fil, -1)


if __name__ == "__main__":
    main()

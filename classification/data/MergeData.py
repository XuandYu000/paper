import argparse

import os
import shutil
import tqdm

def main(datapath: str, mergeName: str) -> None:
    merge_dir = os.path.join(datapath, mergeName)
    training_dir = os.path.join(datapath, "training")
    validation_dir = os.path.join(datapath, "validation")

    if not os.path.exists(merge_dir):
        os.makedirs(merge_dir, exist_ok=True)

        # All files in the training directory and validation directory
        training_files = sorted(os.listdir(training_dir))
        validation_files = sorted(os.listdir(validation_dir))

        last_file_number = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0
        }

        # Copy files from training directory to the merge directory
        for file in tqdm.tqdm(training_files):
            # Get the label from the file name
            label = int(file.split("_")[0])

            file_path = os.path.join(training_dir, file)
            shutil.copy(file_path, merge_dir)
            last_file_number[label] += 1

        for file in tqdm.tqdm(validation_files):
            # Get the label from the file name
            label = int(file.split("_")[0])

            file_number = int(file.split("_")[1].split(".")[0]) + last_file_number[label]
            # Create a new filename with the updated file number
            new_file_name = f"{label}_{file_number}.jpg"
            file_path = os.path.join(validation_dir, file)
            shutil.copy(file_path, os.path.join(merge_dir, new_file_name))

        print(f"Training files number: {len(training_files)}")
        print(f"Validation files number: {len(validation_files)}")
        print(f"Merge files number: {len(os.listdir(merge_dir))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge data files.')
    parser.add_argument("datapath", type=str, help="Path to the data directory")
    parser.add_argument("mergeName", type=str, help="Name of the merged dataSet")
    args = parser.parse_args()

    main(args.datapath, args.mergeName)
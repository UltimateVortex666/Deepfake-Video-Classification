import os

from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from shutil import copyfile

source_folder_1 = './original_sequences/youtube/c40/videos'
source_folder_2 = './manipulated_sequences/NeuralTextures/c40/videos'
dest_folder_1 = './train/1'
dest_folder_2 = './train/0'
dest_folder_3 = './test/1'
dest_folder_4 = './test/0'


# Create destination folders if they don't exist
for folder in [dest_folder_1, dest_folder_2, dest_folder_3, dest_folder_4]:
    os.makedirs(folder, exist_ok=True)

# Check if source folders exist
if not os.path.exists(source_folder_1) or not os.path.exists(source_folder_2):
    print("Source folder not found.")
else:
    # Get the list of files in the source folders
    files_source_1 = [f for f in os.listdir(source_folder_1) if isfile(join(source_folder_1, f))]
    files_source_2 = [f for f in os.listdir(source_folder_2) if isfile(join(source_folder_2, f))]

    # Ensure that there are enough files in both source folders
    if len(files_source_1) >= 860 and len(files_source_2) >= 860:
        # Copy files to destination folders
        for i, j in zip(files_source_1[:860], files_source_2[:860]):
            copyfile(join(source_folder_1, i), join(dest_folder_1, i))
            copyfile(join(source_folder_2, j), join(dest_folder_2, j))

        for i, j in zip(files_source_1[860:], files_source_2[860:]):
            copyfile(join(source_folder_1, i), join(dest_folder_3, i))
            copyfile(join(source_folder_2, j), join(dest_folder_4, j))
    else:
        print("Insufficient files in source folders.")

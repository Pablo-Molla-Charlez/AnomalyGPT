import os
import shutil
import random

# Define the source and target directories
source_dir = '/work/cyh_anomaly/anomalygpt/1AnomalyGPT/cyh_dataset/AeBAD/AeBAD/AeBAD_S'
target_dir = '/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/'

# Define the new structure
new_structure = {'ablation': ['ground_truth', 'test', 'train'],
                 'breakdown': ['ground_truth', 'test', 'train'],
                 'fracture': ['ground_truth', 'test', 'train'],
                 'groove': ['ground_truth', 'test', 'train']}

subfolders = ['background', 'illumination', 'same', 'view']

# Create the new directory structure
"""
for main_folder, sub_folder_list in new_structure.items():
    for sub_folder in sub_folder_list:
        if sub_folder != "train":
            for subsub_folder in subfolders:
                #print(os.path.join(target_dir, "blade_"+main_folder, sub_folder, subsub_folder))
                os.makedirs(os.path.join(target_dir, main_folder, sub_folder, subsub_folder), exist_ok=True)
        else:
            #print(os.path.join(target_dir, "blade_"+main_folder, sub_folder, "good"))
            os.makedirs(os.path.join(target_dir, main_folder, sub_folder, "good"), exist_ok=True)
"""

# Function to copy files from the old structure to the new one
"""
def copy_files(old_dir, new_dir):
    for root, dirs, files in os.walk(old_dir):
        for file in files:
            if file != '.DS_Store':
                old_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, old_dir)
                new_file_path = os.path.join(new_dir, relative_path, file)
                shutil.copy2(old_file_path, new_file_path)
"""
# Copy ground_truth files
"""
for main_folder in new_structure.keys():
    for subsub_folder in subfolders:
        old_path = os.path.join(source_dir, 'ground_truth', main_folder, subsub_folder)
        new_path = os.path.join(target_dir, "blade_"+main_folder, 'ground_truth', subsub_folder)
        print("old:", old_path, "\nnew: ", new_path, "\n\n")
        copy_files(old_path, new_path)
"""

# Copy test files
"""
for main_folder in new_structure.keys():
    for subsub_folder in subfolders:
        old_path = os.path.join(source_dir, 'test', main_folder, subsub_folder)
        new_path = os.path.join(target_dir, "blade_"+main_folder, 'test', subsub_folder)
        print("old:", old_path, "\nnew: ", new_path, "\n\n")
        copy_files(old_path, new_path)
"""

# Define source and target directories
"""
source_folders = [
    "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/cyh_dataset/AeBAD/AeBAD/AeBAD_S/train/good/background",
    "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/cyh_dataset/AeBAD/AeBAD/AeBAD_S/train/good/illumination",
    "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/cyh_dataset/AeBAD/AeBAD/AeBAD_S/train/good/view"]

target_folders = [
    "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/blade_ablation/train/good",
    "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/blade_breakdown/train/good",
    "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/blade_fracture/train/good",
    "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/blade_groove/train/good"]
"""

source_folders = "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/ablation/train/good/"

target_folders = [
    "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/ablation/test/good",
    "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/breakdown/test/good",
    "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/fracture/test/good",
    "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/groove/test/good"]

# Function to copy a percentage of files from source to target
"""
def copy_random_files(source_folder, target_folders, percentage):
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    num_files_to_copy = int(len(files) * percentage)
    print("Files to copy:", num_files_to_copy)
    files_to_copy = random.sample(files, num_files_to_copy)

    
    for file in files_to_copy:
        for target_folder in target_folders:
            shutil.copy2(os.path.join(source_folder, file), target_folder)
"""    

# Creation of folder train for ablation, breakdown, fracture and groove
# Copy 70% of the files from each source folder to each target folder
"""
for source_folder in source_folders:
    # Each category train has the same random 70%
    copy_random_files(source_folder, target_folders, 0.7)
"""

# Creation of folder test/good for ablation, breakdown, fracture and groove
#copy_random_files(source_folders, target_folders, 0.3)


def rename_files(directory):
    files = sorted(os.listdir(directory))
    
    #print(files, "\n")

    for idx, filename in enumerate(files):
        if filename.endswith(".png"):
            new_name = f"{idx:03d}.png"
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_name)
            #print("SRC:", src)
            #print("DST:", dst, "\n")
            os.rename(src, dst)
            print(f"Renamed {filename} to {new_name}")

# Set the path to the directory
directory = "/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/fracture/train/good"

# Call the function to rename files
rename_files(directory)

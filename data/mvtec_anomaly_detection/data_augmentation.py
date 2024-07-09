import imgaug.augmenters as iaa
import cv2
import os
import numpy as np
from collections import defaultdict


# Define augmentation sequence
dict_aug = {'bottle': iaa.Sequential([
                            iaa.Sometimes(0.8, iaa.Affine(rotate=180)), # Rotation of 180º with a probability of 80%
                            iaa.Sometimes(0.6, iaa.Affine(rotate=90)), # Rotation of 90º with a probability of 60%
                            iaa.Sometimes(0.5, iaa.Fliplr(1)), # Horizontal flip with a probability of 50%
                            iaa.Flipud(1) # Vertical flip
                            ]), 
            'cable':iaa.Sequential([
                            iaa.Sometimes(0.8, iaa.Affine(rotate=180)), # Rotation of 180º with a probability of 80%
                            iaa.Sometimes(0.6, iaa.Affine(rotate=90)), # Rotation of 90º with a probability of 60%
                            iaa.Fliplr(1), # Horizontal flip
                            iaa.Sometimes(0.7, iaa.Multiply((0.9, 1.1))) # Brightness tuning downgrade/upgrade between 90%/110% from the original brightness with a probability of 70%
                            ]),
            'capsule': iaa.Sequential([
                            iaa.Affine(rotate=270), # Rotation of 270º
                            iaa.Sometimes(0.5, iaa.Affine(rotate=180)), # Rotation of 180º with a probability of 50%
                            iaa.Sometimes(0.7, iaa.Multiply((0.9, 1.1))) # Brightness tuning downgrade/upgrade between 90%/110% from the original brightness with a probability of 70%
                            ]),
            'carpet': iaa.Sequential([
                            iaa.Sometimes(0.5, iaa.Affine(rotate=180)), # Rotation of 180º with a probability of 50%
                            iaa.Sometimes(0.5, iaa.Affine(rotate=270)), # Rotation of 270º with a probability of 50%
                            iaa.Sometimes(0.5, iaa.Flipud(1)), # Vertical flip with a probability of 50%
                            iaa.Sometimes(0.7, iaa.Multiply((0.9, 1.1)))]), # Brightness tuning downgrade/upgrade between 90%/110% from the original brightness with a probability of 70%
            'grid': iaa.Sequential([
                            iaa.Sometimes(0.5, iaa.Affine(rotate=180)), # Rotation of 180º with a probability of 50%
                            iaa.Sometimes(0.5, iaa.Affine(rotate=270)), # Rotation of 270º with a probability of 50%
                            iaa.Sometimes(0.5, iaa.Flipud(1)), # Vertical flip with a probability of 50%
                            iaa.Sometimes(0.7, iaa.Multiply((0.9, 1.1)))]), # Brightness tuning downgrade/upgrade between 90%/110% from the original brightness with a probability of 70%
            'hazelnut': iaa.Sequential([
                            iaa.Sometimes(0.5, iaa.Affine(rotate=90)), # Rotation of 90º with a probability of 50%
                            iaa.Sometimes(0.5, iaa.Affine(rotate=270)), # Rotation of 270º with a probability of 50%
                            iaa.Sometimes(0.5, iaa.Fliplr(1)), # Horizontal flip with a probability of 50%
                            iaa.Sometimes(0.7, iaa.Multiply((0.9, 1.1)))]), # Brightness tuning downgrade/upgrade between 90%/110% from the original brightness with a probability of 70%
            'leather': iaa.Sequential([
                            iaa.Sometimes(0.5, iaa.Affine(rotate=90)), # Rotation of 90º with a probability of 50%
                            iaa.Sometimes(0.5, iaa.Affine(rotate=180)), # Rotation of 180º with a probability of 50%
                            iaa.Flipud(1), # Vertical flip
                            iaa.Sometimes(0.7, iaa.Multiply((0.9, 1.1)))]), # Brightness tuning downgrade/upgrade between 90%/110% from the original brightness with a probability of 70%
            'metal_nut': iaa.Sequential([
                            iaa.Sometimes(0.5, iaa.Affine(rotate=90)), # Rotation of 90º with a probability of 50%
                            iaa.Sometimes(0.5, iaa.Affine(rotate=180)), # Rotation of 180º with a probability of 50%
                            iaa.Flipud(1)]), # Vertical flip
            'pill': iaa.Sequential([
                            iaa.Affine(rotate=(-35,35)), # Rotation between -35º and 35º
                            iaa.Affine(scale=(1.0, 1.2)), # Zoom into the original image between 0-20%
                            iaa.Sometimes(0.6, iaa.Fliplr(1)), 
                            iaa.Sometimes(0.7, iaa.Flipud(1))]),
            'screw': iaa.Sequential([iaa.Affine(rotate=(-35,35)), iaa.Affine(scale=(1.2, 1.5)), iaa.Sometimes(0.6, iaa.Fliplr(1)), iaa.Sometimes(0.6, iaa.Flipud(1))]),
            'tile': iaa.Sequential([iaa.Affine(rotate=(-35,35)), iaa.Affine(scale=(1.2, 1.4))]), 
            'toothbrush': iaa.Sequential([iaa.Flipud(1), iaa.Affine(rotate=(-20,20))]), 
            'transistor': iaa.Sequential([iaa.Affine(rotate=(-15,15)), iaa.Sometimes(0.6, iaa.Flipud(1))]), 
            'wood': iaa.Sequential([iaa.Affine(rotate=(-50,50)), iaa.Affine(scale=(1.5, 1.8)), iaa.Sometimes(0.7, iaa.Flipud(1))]), 
            'zipper': iaa.Sequential([iaa.Affine(rotate=(-15,15)), iaa.Affine(scale=(1.4, 1.6)), iaa.Flipud(1)])
            }

# Data Augmentaiton only for the "train/good" samples
def augment_images(image_dir, output_dir, target_samples):

    # To create directory if doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_counts = defaultdict(int)
    list_img = sorted(os.listdir(image_dir))

    for class_name in list_img:
        if (class_name[-2:] != "xt") and (class_name[-2:] != "py"):
            if class_name in dict_aug.keys():
                print(f"--------------------------------- CLASS {class_name} initiated ---------------------------------")
                class_path = os.path.join(image_dir, class_name, "train/good")
                output_class_path = os.path.join(output_dir, class_name, "train/good")

                #print("Class_Path:", class_path)
                #print("Output_Path:", output_class_path)
                image_files = [f for f in sorted(os.listdir(class_path)) if os.path.isfile(os.path.join(class_path, f))]
                num_existing_images = len(image_files)

                print("Existing_Images:", num_existing_images)

                
                for idx, image_file in enumerate(image_files):
                    image_path = os.path.join(class_path, image_file)
                    print("Original:", image_path)
                    image = cv2.imread(image_path)
                    print("Nº of images intented to create:", round(target_samples / num_existing_images))

                    for i in range(round(target_samples / num_existing_images)): # 1000//60 = 16
                        #print("DICT_AUG:", dict_aug[class_name])
                        augmented_image = dict_aug[class_name](image=image)
                        augmented_image_path = os.path.join(output_class_path, f"{class_name}_{idx}_{i}.png")
                        cv2.imwrite(augmented_image_path, augmented_image)
                        print("New:",augmented_image_path)
                        class_counts[class_name] += 1
                
                        new_image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
                        new_num_existing_images = len(new_image_files)

        for category, count in class_counts.items():
            if class_name == category:
                print("New_Existing_Images:", new_num_existing_images)
                print(f"--------------------------------- CLASS {category}: {count} augmented images created ---------------\n\n")

# Set paths and target number of samples per class
image_dir = '/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/'
output_dir = '/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection/'
target_samples = 1000  # Adjust this number as needed

augment_images(image_dir, output_dir, target_samples)


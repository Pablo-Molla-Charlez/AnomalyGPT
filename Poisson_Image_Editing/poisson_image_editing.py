"""Poisson image editing.

"""
import os
import shutil
import numpy as np
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve
import imgaug.augmenters as iaa
from os import path

def laplacian_matrix(n, m):
    """Generate the Poisson matrix. 

    Refer to: 
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation

    Note: it's the transpose of the wiki's matrix 
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A


def poisson_edit(source, target, mask, offset):
    """The poisson blending function. 

    Refer to: 
    Perez et. al., "Poisson Image Editing", 2003.
    """

    # Assume: 
    # target is not smaller than source.
    # shape of mask is same as shape of target.
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min
        
    M = np.float32([[1,0,offset[0]],[0,1,offset[1]]])
    source = cv2.warpAffine(source,M,(x_range,y_range))
        
    mask = mask[y_min:y_max, x_min:x_max]    
    mask[mask != 0] = 1
    #mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    
    mat_A = laplacian_matrix(y_range, x_range)

    # for \Delta g
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity    
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0

    # corners
    # mask[0, 0]
    # mask[0, y_range-1]
    # mask[x_range-1, 0]
    # mask[x_range-1, y_range-1]

    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()    
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()        

        # inside the mask:
        # \Delta f = div v = \Delta g       
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        
        x = spsolve(mat_A, mat_b)
        
        x = x.reshape((y_range, x_range))
        
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')  

        target[y_min:y_max, x_min:x_max, channel] = x

    return target

def main():
    # Define augmentation sequence
    aug = iaa.Sequential([  
            #iaa.Fliplr(1),  # horizontal flips with 100% probability
            #iaa.Sometimes(0.8, iaa.Affine(rotate=180)),
            #iaa.Sometimes(0.6, iaa.Affine(rotate=90)),  # rotate by -20 to +20 degrees
            iaa.Affine(rotate=270),
            #iaa.Affine(scale=(1.0, 1.2)), # Zoom into image from 20% to 50%
            #iaa.Affine(scale=(0.8, 1.0)) # zoom out from 100% to 80%
            #iaa.Sometimes(0.5, iaa.Fliplr(1)),
            #iaa.Flipud(1), # vertical flip 
            #iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),  # translate
            #iaa.Sometimes(0.7, iaa.Multiply((0.8, 1.4))),  # change brightness
            #iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images
            ])

    base_dir = '/work/cyh_anomaly/anomalygpt/1AnomalyGPT/data/mvtec_anomaly_detection'
    
    # CLASS_NAMES does not consider the 4 new categories as they are highly populated
    CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 
                   'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    
    

    for class_name in CLASS_NAMES:
        print(f"-------------- CLASS {class_name}: STARTED --------------")
        ground_truth_dir = path.join(base_dir, class_name, 'ground_truth')
        test_dir = path.join(base_dir, class_name, 'test')
        train_dir = path.join(base_dir, class_name, 'train/good')

        for root, dirs, files in os.walk(ground_truth_dir):
            for subfolder in dirs:
                print(f"-------------- SUBFOLDER {subfolder} --------------")
                img_counter = 0
                gt_folder = path.join(ground_truth_dir, subfolder)
                test_folder = path.join(test_dir, subfolder)
                train_folder = train_dir

                if not path.exists(test_folder):
                    continue

                gt_images = sorted(os.listdir(gt_folder))
                test_images = sorted(os.listdir(test_folder))
                train_images = sorted(os.listdir(train_folder))

                for i in range(len(test_images)):
                    source_img_path = path.join(test_folder, test_images[i])
                    mask_img_path = path.join(gt_folder, gt_images[i])
                    target_img_path = path.join(train_folder, train_images[i])

                    # Loading source, target and both mask images
                    source = cv2.imread(source_img_path)
                    target = cv2.imread(target_img_path)
                    
                    # For applying the poisson dist. image editing we need the grayscale
                    mask_poisson = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
                    
                    # For normal data augmentation operations, we don't need grayscale
                    mask_augmentation = cv2.imread(mask_img_path)

                    offset = (0, 0)
                    result_poisson = poisson_edit(source, target, mask_poisson, offset)

                    new_anomaly_img_name = f'new_anomaly_{img_counter:03d}.png'
                    new_anomaly_mask_name = f'new_anomaly_mask_{img_counter:03d}.png'
                    
                    result_augmentation = aug(image=result_poisson)
                    result_augmentation_mask = aug(image=mask_augmentation)
                    
                    # New Anomaly Image for testing
                    cv2.imwrite(path.join(test_folder, new_anomaly_img_name), result_augmentation)
                    #print("ANOMALY:", path.join(test_folder, new_anomaly_img_name))
                    
                    # New Mask for previous image
                    cv2.imwrite(path.join(gt_folder, new_anomaly_mask_name), result_augmentation_mask)
                    #print("MASK:", path.join(gt_folder, new_anomaly_mask_name), "\n")
                    img_counter += 1
        print(f"-------------- CLASS {class_name}: ENDED --------------")
        


if __name__ == "__main__":
    main()



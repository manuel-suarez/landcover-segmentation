import os
import cv2
import numpy as np
import glob

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
import random

# Configure dirs
home_dir = os.path.expanduser('~')
data_dir = os.path.join(home_dir, 'data')
work_dir = os.path.join(data_dir, 'landcover-ai')
img_dir = os.path.join(work_dir, 'images')
mask_dir = os.path.join(work_dir, 'masks')

temp_img = cv2.imread(os.path.join(img_dir, 'M-34-51-C-d-4-1.tif'))
plt.imshow(temp_img[:,:,2])
plt.savefig("figure01.png")
plt.close()
temp_mask = cv2.imread(os.path.join(mask_dir, 'M-34-51-C-d-4-1.tif'))
labels, count = np.unique(temp_mask[:,:,0], return_counts=True)
print("Labels are: ", labels, " and the counts are: ", count)

patch_size = 256
for path, subdirs, files in os.walk(img_dir):
    # print(path)
    dirname = path.split(os.path.sep)[-1]
    # print(dirname)
    images = os.listdir(path)  # List of all image names in this subdirectory
    # print(images)
    for i, image_name in enumerate(images):
        if image_name.endswith(".tif"):
            # print(image_name)
            image = cv2.imread(path + "/" + image_name, 1)  # Read each image as BGR
            SIZE_X = (image.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
            SIZE_Y = (image.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
            image = Image.fromarray(image)
            image = image.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
            # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
            image = np.array(image)

            # Extract patches from each image
            print("Now patchifying image:", path + "/" + image_name)
            patches_img = patchify(image, (256, 256, 3), step=256)  # Step=256 for 256 patches means no overlap

            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, :, :]
                    # single_patch_img = (single_patch_img.astype('float32')) / 255. #We will preprocess using one of the backbones
                    single_patch_img = single_patch_img[
                        0]  # Drop the extra unecessary dimension that patchify adds.

                    cv2.imwrite(os.path.join(work_dir, "256_patches", "images",
                                             image_name + "patch_" + str(i) + str(j) + ".tif"),
                                single_patch_img)
                    # image_dataset.append(single_patch_img)

for path, subdirs, files in os.walk(mask_dir):
    # print(path)
    dirname = path.split(os.path.sep)[-1]

    masks = os.listdir(path)  # List of all image names in this subdirectory
    for i, mask_name in enumerate(masks):
        if mask_name.endswith(".tif"):
            mask = cv2.imread(path + "/" + mask_name,
                              0)  # Read each image as Grey (or color but remember to map each color to an integer)
            SIZE_X = (mask.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
            SIZE_Y = (mask.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
            mask = Image.fromarray(mask)
            mask = mask.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
            # mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
            mask = np.array(mask)

            # Extract patches from each image
            print("Now patchifying mask:", path + "/" + mask_name)
            patches_mask = patchify(mask, (256, 256), step=256)  # Step=256 for 256 patches means no overlap

            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    single_patch_mask = patches_mask[i, j, :, :]
                    # single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                    # single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.
                    cv2.imwrite(os.path.join(work_dir, "256_patches", "masks",
                                             mask_name + "patch_" + str(i) + str(j) + ".tif"),
                                single_patch_mask)

train_img_dir = os.path.join(work_dir, "256_patches", "images")
train_mask_dir = os.path.join(work_dir, "256_patches", "masks")

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

num_images = len(os.listdir(train_img_dir))


img_num = random.randint(0, num_images-1)

img_for_plot = cv2.imread(os.path.join(train_img_dir, img_list[img_num]), 1)
img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)

mask_for_plot =cv2.imread(os.path.join(train_mask_dir, msk_list[img_num]), 0)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_for_plot)
plt.title('Image')
plt.subplot(122)
plt.imshow(mask_for_plot, cmap='gray')
plt.title('Mask')
plt.savefig("figure02.png")
plt.close()

# Now, let us copy images and masks with real information to a new folder.
# real information = if mask has decent amount of labels other than 0.

useless = 0  # Useless image counter
for img in range(len(img_list)):  # Using t1_list as all lists are of same size
    img_name = img_list[img]
    mask_name = msk_list[img]
    print("Now preparing image and masks number: ", img)

    temp_image = cv2.imread(os.path.join(train_img_dir, img_list[img]), 1)
    temp_mask = cv2.imread(os.path.join(train_mask_dir, msk_list[img]), 0)
    # temp_mask=temp_mask.astype(np.uint8)

    val, counts = np.unique(temp_mask, return_counts=True)

    if (1 - (counts[0] / counts.sum())) > 0.05:  # At least 5% useful area with labels that are not 0
        print("Save Me")
        cv2.imwrite(os.path.join(work_dir, "256_patches", "images_with_useful_info/images/" + img_name), temp_image)
        cv2.imwrite(os.path.join(work_dir, "256_patches", "images_with_useful_info/masks/" + mask_name), temp_mask)

    else:
        print("I am useless")
        useless += 1

print("Total useful images are: ", len(img_list) - useless)  # 20,075
print("Total useless images are: ", useless)  # 21,571
###############################################################
#Now split the data into training, validation and testing.

"""
Code for splitting folder into train, test, and val.
Once the new folders are created rename them and arrange in the format below to be used
for semantic segmentation using data generators. 

pip install split-folders
"""
import splitfolders  # or import split_folders

input_folder = os.path.join(work_dir, "256_patches", "images_with_useful_info")
output_folder = os.path.join(work_dir, 'results')
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None) # default values

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
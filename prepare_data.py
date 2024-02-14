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
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import argparse
# Root directory of the project

MODEL_PATH = "../Mask_RCNN/mask_rcnn_coco.h5"

# Import Mask RCNN
ROOT_DIR = os.path.abspath("./Mask_RCNN")
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize, config

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = MODEL_PATH
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = 'coco'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_NMS_THRESHOLD = 0.8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
print(COCO_MODEL_PATH)
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# image = skimage.io.imread("../test.jpg")
# # Run detection
# results = model.detect([image], verbose=1)
#
# # Visualize results
# r = results[0]
#
# print("main visualize result")
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])

def car_mask(image):

    car_image = np.zeros_like(image)
    results = model.detect([image], verbose=1)
    for r in results:

        class_id = r['class_ids']
        indices = np.where(class_id == 3)
        cars = r['rois'][indices]

        for car in cars:
            y1, x1, y2, x2 = car
            car_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]

        # print("main visualize result")
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                         class_names, r['scores'])

    return car_image

# image = skimage.io.imread("../test.jpg")
# # Run detection
# car = car_mask(image)
# print(car.shape)
# skimage.io.imsave("../result.jpg", car)
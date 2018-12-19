import os
from PIL import Image, ImageFile
import random
import numpy as np
import json
import cv2
from utils import data_util

ImageFile.LOAD_TRUNCATED_IMAGES = True

labelColor = {'白实线':60, '黄实线':60, '停止线':75,
                '左转车道': 90, '右转车道':105, '直行车道':120,
              '右转直行车道':135, '左转直行车道':150,
              '左转直行信号灯':165, '右转直行信号灯':180,
              '直行信号灯':195, '左转信号灯':210, '右转信号灯':225,}

def load_label(filepath):

    fname, tmps, width, height, depth, polygons = data_util.paser_xml(filepath)
    mask = np.zeros((height, width), dtype=np.uint8)

    for label, polygon in polygons:
        # print(b.shape, points.shape)
        polygon = polygon[np.newaxis, :]
        color = labelColor.get(label)
        cv2.fillPoly(mask, polygon, color)

    return tmps, width, height, mask


def load_data(imagepath, xmlpath, name):
    items = name.split("_")
    filepath = os.path.join(xmlpath, items[2] + ".xml")
    tmps, mask_w, mask_h, mask = load_label(filepath)

    filepath = os.path.join(imagepath, name)
    image = Image.open(filepath).convert("RGB")

    image = np.asarray(image)
    print(np.array(image, dtype=np.uint8), image)
    array = data_util.image_split(np.array(image), tmps, (mask_w, mask_h))
    array = np.stack(array)

    mask = mask[:, :, np.newaxis]
    mask = np.tile(mask, (array.shape[0], 1, 1, 1))

    array = np.concatenate((array, mask), -1)

    label = int(items[0])
    return array, label


if __name__ == "__main__":

    array, label = load_data("../../data/1345/1_1345", "../../mark",
        "1_1345_130929000000020012C_¼½J086SW_02_20181002072755_1309290600064238.jpg")

    # array, label = load_data("../../data/1345/1_1345", "../../mark",
    #     "1_1345_130581000000010160C_¼½E1W059_02_20180909165605_1305810300087389.jpg")

    for i, image in enumerate(array):
        cv2.imwrite("result_{}.png".format(i), image[...,:-1])
        cv2.imwrite("label.png", image[...,-1])




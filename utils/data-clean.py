#encoding=utf-8

import cv2
from PIL import Image
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from utils import data_util, find

def to_unicode(path):
    upath = path.encode('gbk').decode(errors='ignore')
    return upath

def xml_des(path):
    flist = os.listdir(path)

    for file in flist:
        # print(file)
        result = data_util.paser_xml(os.path.join(path, file))
        fname, tmps, width, height, depth, polygons = result
        print(file, tmps.strip())

def line_split(image):

    img = image.convert("P")
    img = np.array(img)
    image = np.array(image)
    w, h = img.shape

    print(w, h)

    cv2.line(image, (h//2, 0), (h//2, w), color=(0,0,255), thickness=10)

    gradient = np.gradient(img, axis=0)
    dy = np.sum(np.abs(gradient), axis=1)
    print(dy.shape)

    indices = np.argpartition(dy, range(-10,0))[-10:]
    indices = sorted(indices)
    print(indices)

    ids = [1,6,-2,-1]
    for id in ids:
        index = indices[id]
        cv2.line(image, (0, index), (h, index), color=(0,0,255), thickness=10)
    return image

def image_split(img, mode, shape):
    h = img.shape[0]
    w = img.shape[1]
    result = []

    if mode == "2*2":
        # tuple(start, width)
        h_ratio = [(0, 0.5), (0.5, 0.5)]
        v_ratio = [(0, 0.5), (0.5, 0.5)]
    if mode == "1*2":
        # tuple(start, width)
        h_ratio = [(0, 1.0)]
        v_ratio = [(0, 0.5), (0.5, 0.5)]
    if mode == "1*3":
        # tuple(start, width)
        h_ratio = [(0, 1.0)]
        v_ratio = [(0, 0.333), (0.333, 0.333), (0.667, 0.333)]

    for hr in h_ratio:
        for vr in v_ratio:
            hup = int(h*hr[0])
            hdown = hup+int(h*hr[1])
            rleft = int(w*vr[0])
            rright = rleft+int(w*vr[1])
            crop = img[hup:hdown, rleft:rright]
            crop = cv2.resize(crop, shape)
            result.append(crop)

    if mode == "2*2":
        carid = find.find_car(result)
        result.pop(carid)

    if mode == "1*2":
        result.append(np.zeros_like(result[0]))
    if mode == "1*3":
        carid = find.find_car(result)
        result.pop(carid)
        result.append(np.zeros_like(result[0]))

    return result, carid

if __name__ == "__main__":

    fileid = 3
    img = np.array(Image.open("../../report/image/{}.jpg".format(fileid)))
    result, id = image_split(np.array(img), mode="1*3", shape=(500,400))
    print(id)
    h = img.shape[0]
    w = img.shape[1]
    img[0:h,2*w//3:w] = 0
    cv2.imwrite("../../report/image/{}_0.jpg".format(fileid), img)


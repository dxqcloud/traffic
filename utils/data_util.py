#-*- coding: utf-8 -*-

import xml.etree.cElementTree as ET
import numpy as np
import cv2
import os
from utils import find
from sklearn.model_selection import train_test_split

labelColor = {'白实线':60, '黄实线':60, '停止线':75,
                '左转车道': 90, '右转车道':105, '直行车道':120,
              '右转直行车道':135, '左转直行车道':150,
              '左转直行信号灯':165, '右转直行信号灯':180,
              '直行信号灯':195, '左转信号灯':210, '右转信号灯':225,}

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

    return result

def paser_xml(file):

    tree = ET.parse(file)
    root = tree.getroot()

    fname = root.find('filename').text
    tmps = root.find('mode').text.strip()

    width = 0
    height = 0
    depth = 0
    size_node = root.find('size')
    for child in size_node:
        if child.tag == 'width':
            width = int(child.text)
        elif child.tag == 'height':
            height = int(child.text)
        else:
            depth = int(child.text)

    polygons = []
    for item in root.iter('object'):
        name = item.find('name').text
        box = item.find('bndbox')
        poly = item.find('polygon')
        polygon = None

        if box != None:
            xmin = int(box.find('xmin').text)
            xmax= int(box.find('xmax').text)
            ymin = int(box.find('ymin').text)
            ymax = int(box.find('ymax').text)
            polygon = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
        elif poly != None:
            x = []
            y = []
            for child in poly:
                tag = child.tag
                if tag[0] == 'x':
                    x.append(int(child.text))
                else:
                    y.append(int(child.text))
            polygon = np.stack([x,y],axis=1)
        polygons.append((name, polygon))

    return (fname, tmps, width, height, depth, polygons)

def SplitData(path, mode, xlist, test_size=0.3, seed = 0):

    class_path = os.listdir(path)

    train_names = []
    test_names= []
    all = []

    for class_type in class_path:
        names = []
        for file in os.listdir(os.path.join(path, class_type)):
            # print(file)
            items = file.split("_")
            if items[2]+".xml" in xlist:
                names.append(os.path.join(class_type, file))

        # print(names.shape)
        train, test = train_test_split(names, test_size=test_size, random_state=seed)
        train_names.extend(train)
        test_names.extend(test)
        all.extend(train)
        all.extend(test)
        np.random.shuffle(train_names)

    if mode == "train":
        return train_names, test_names
    else:
        return all


if __name__ == "__main__":

    result = paser_xml('../../test/xml/371594000000010052.xml')
    fname, tmps, width, height, depth, polygons = result
    mask = np.zeros((height, width), dtype=np.uint8)

    for label, polygon in polygons:
        # print(b.shape, points.shape)
        polygon = polygon[np.newaxis,:]
        color = labelColor.get(label)
        cv2.fillPoly(mask, polygon, color)
        print(label, color)
    cv2.imwrite("mask.png", mask)
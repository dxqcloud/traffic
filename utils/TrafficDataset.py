#encoding=utf-8

import os
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import numpy as np
from PIL import Image, ImageFile
from utils import data_util
from torchvision.transforms import ToTensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# labelColor = {'leftline': 10, 'rightline':30, 'straitline':50,
#               'rightstraitline':70, 'leftstraitline':90,
#                'whiteline':110, 'stopline':130,
#               'leftstraitlight':150, 'rightstraitlight':170,
#                'leftlight':190, 'straitlight':210,'rightlight':230,}

labelColor = {'白实线':60, '黄实线':60, '停止线':75,
                '左转车道': 90, '右转车道':105, '直行车道':120,
              '右转直行车道':135, '左转直行车道':150,
              '左转直行信号灯':165, '右转直行信号灯':180,
              '直行信号灯':195, '左转信号灯':210, '右转信号灯':225,}

toTensor = ToTensor()

def typeOneHot(type):
    typelist = {"1625":0, "1208":1, "1345":2}
    vec = np.zeros(3)
    id = typelist.get(type)
    vec[id] = 1
    return vec


class TrafficDataSet(Dataset):
    def __init__(self, imagepath, xmlpath, names, width, height,
                 mode="train", transform=None):

        self.imagepath = imagepath
        self.xmlpath = xmlpath
        self.width = width
        self.height = height
        self.shape = (width, height)
        self.transform = transform
        self.mode = mode
        self.names = names

    def load_label(self, file):
        filepath = os.path.join(self.xmlpath, file)

        fname, tmps, width, height, depth, polygons = data_util.paser_xml(filepath)
        mask = np.zeros((height, width), dtype=np.uint8)

        for label, polygon in polygons:
            # print(b.shape, points.shape)
            polygon = polygon[np.newaxis, :]
            color = labelColor.get(label)
            cv2.fillPoly(mask, polygon, color)

        return tmps, width, height, mask

    def load_data(self, file_path):

        file_name = os.path.basename(file_path)
        items = file_name.split("_")

        if self.mode == "predict":
            scene = items[1]
        else:
            scene = items[2]

        tmps, mask_w, mask_h, mask = self.load_label(scene+".xml")
        mask = cv2.resize(mask, (self.width, self.height))
        mask = mask[...,np.newaxis]

        filepath = os.path.join(self.imagepath, file_path)
        image = Image.open(filepath)
        image = np.array(image)

        array = data_util.image_split(image, tmps, (self.width, self.height))
        label_image = []

        mask = toTensor(mask)
        for timage in array:
            # print(timage.shape, mask.shape)
            image = Image.fromarray(timage)
            if self.transform:
                image = self.transform(image)

            image = toTensor(image)
            image = torch.cat((image, mask))
            label_image.append(image)

        label_image = torch.stack(label_image)
        # mask = mask[:,:,np.newaxis]
        # mask = np.tile(mask,(array.shape[0],1,1,1))
        # array = np.concatenate((array, mask), -1)

        return label_image


    def __getitem__(self, index):
        file = self.names[index]
        # print(file)

        x = self.load_data(file)

        items = file.split("_")
        if self.mode == "predict":
            type = items[0][:-1]
            scene = items[1]
            return x, typeOneHot(type)
        else:
            y = int(items[0])
            type = items[1][:-1]
            scene = items[2]
            return x, typeOneHot(type), y


    def __len__(self):
        return len(self.names)

ipath = "/home/traffic/data/train/"
xpath = "/home/traffic/data/mask/"

if __name__ == "__main__":

    xlist = os.path.listdir(xpath)
    train_names, test_names = data_util.SplitData(ipath, "train", xlist)

    dataset = TrafficDataSet(ipath, xpath, train_names, 1000, 800)
    dataloader = DataLoader(dataset, 1, num_workers=2, shuffle=True)
    datait = iter(dataloader)
    # count = 0
    # for batch, id in datait:
    #     count += 1
    #     print(batch.shape, count)

    image, _ = next(datait)
    print(image[0, 0, 3].shape)


    # # plt.imshow(image,origin=(0,0))
    # plt.imshow(label)
    #
    # plt.show()
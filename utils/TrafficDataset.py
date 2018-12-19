#encoding=utf-8

import os
from torch.utils.data import DataLoader, Dataset
import cv2
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

class TrafficDataSet(Dataset):
    def __init__(self, imagepath, xmlpath, names, width, height, transform=None):
        self.imagepath = imagepath
        self.xmlpath = xmlpath
        self.width = width
        self.height = height
        self.shape = (width, height)
        self.transform = transform

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
        tmps, mask_w, mask_h, mask = self.load_label(items[2]+".xml")
        mask = cv2.resize(mask, (self.width, self.height))

        filepath = os.path.join(self.imagepath, file_path)
        image = Image.open(filepath)
        image = np.array(image)

        array = data_util.image_split(image, tmps, (self.width, self.height))

        array = np.stack(array)

        mask = mask[:,:,np.newaxis]
        mask = np.tile(mask,(array.shape[0],1,1,1))
        array = np.concatenate((array, mask), -1)

        label = int(items[0])
        return array, label

    def __getitem__(self, index):
        file = self.names[index]
        # print(file)
        x, y = self.load_data(file)
        x = np.transpose(x, (0, 3, 1, 2)) / 255.0

        if self.transform:
            x = self.transform(x)

        return x, y

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
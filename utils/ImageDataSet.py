#encoding=utf-8

import os
import torch
import traceback
from torch.utils.data import DataLoader, Dataset
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split


# labelColor = {'leftline': 10, 'rightline':30, 'straitline':50,
#               'rightstraitline':70, 'leftstraitline':90,
#                'whiteline':110, 'stopline':130,
#               'leftstraitlight':150, 'rightstraitlight':170,
#                'leftlight':190, 'straitlight':210,'rightlight':230,}

class_label = {"1208":0, "1625":1}

def SplitData(path, mode, test_size=0.3, seed = 0):

    class_path = os.listdir(path)

    train_names = []
    test_names= []
    all = []

    for class_type in class_path:
        names = set()
        for file in os.listdir(os.path.join(path, class_type)):
            # print(file)
            name, _ = file.split("_")
            names.add("{}_{}".format(class_type, name))

        names = sorted(names)
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


class ImageDataSet(Dataset):
    def __init__(self, path, names, width, height, transform=None):
        self.path = path
        self.width = width
        self.heigth = height
        self.shape = (width, height)
        self.names = names
        self.transform = transform

    def load_data(self, name):
        array = []

        type, true_name = name.split("_")
        for i in range(1,4):

            file = "{}_{}.jpg".format(true_name, str(i).zfill(2))
            filePath = os.path.join(self.path, type, file)

            image = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1) / 255.0

            if self.transform:
                image = self.transform(image)
            # print(file)
            img_height, img_width, _ = image.shape
            image = cv2.resize(image, self.shape)
            # print(self.shape, image.shape)
            array.append(image)

        array = np.stack(array)

        assert type in class_label
        label = class_label[type]
        return array, label

    def __getitem__(self, index):
        file = self.names[index]
        # print(file)
        x, y = self.load_data(file)
        x = np.transpose(x, (0, 3, 1, 2))

        return torch.FloatTensor(x), y

    def __len__(self):
        return len(self.names)


if __name__ == "__main__":

    train_names, test_names= SplitData("../../raw-data/train", "test")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
       [transforms.RandomRotation(10), normalize]
    )

    test_names = SplitData("../../raw-data/test", "test")

    dataset = ImageDataSet("../../raw-data/train", train_names, 500, 400, transform)
    train_loader = DataLoader(dataset, 2, num_workers=2, shuffle=True)

    dataset = ImageDataSet("../../raw-data/test", test_names, 500, 400)
    test_loader = DataLoader(dataset, 2, num_workers=2, shuffle=True)

    # count = 0
    # for batch, id in datait:
    #     count += 1
    #     print(batch.shape, count)

    print(len(train_loader), len(test_loader))
    print(train_names)
    for x, y in train_loader:
        print(x.shape, y)
        # print(y)
    # # plt.imshow(image,origin=(0,0))
    # plt.imshow(label)
    #
    # plt.show()
#encoding=utf-8
import os
import torch.nn as nn
import torch
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import argparse

from utils.TrafficDataset import TrafficDataSet
from utils import data_util

paser = argparse.ArgumentParser()

class Model(nn.Module):

    def __init__(self, num_class):
        super(Model, self).__init__()
        self.num_class = num_class
        resnet = models.resnet18(True)
        feature_size = 107520

        resnet = nn.Sequential(*list(resnet.children())[:-1])

        # resnet.fc = nn.LeakyReLU(0.1)

        for param in resnet.parameters():
            param.requires_grad = False

        self.resnet = resnet
        self.fc1 = nn.Linear(feature_size, 1024)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(1024, num_class)

        # print(self.resnet)

    def forward(self, input):
        feature = []
        # input = input[0]
        # print(input.size())
        for i in range(3):
            x = self.resnet(input[:,i,:,:,:])
            x = x.view(x.size(0), -1)
            feature.append(x)
        feature = torch.cat(feature,1)
        # print(feature.size())
        out = self.fc1(feature)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def train(train_data, test_data, model, args):
    lossfunc = nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # print("training samples:", len(train_data))
    for epoch in range(args.epochs):
        batch = 0
        for x, y in train_data:

            optimzer.zero_grad()
            x = x[:, :, 0:3, :, :]
            x = x.type("torch.FloatTensor")
            if args.use_gpu:
                x = x.cuda()
                y = y.cuda()
            out = model.forward(x)
            loss = lossfunc(out, y)
            loss.backward()
            optimzer.step()
            print("loss:", loss.item())
            batch += 1
            # if batch > 2:
            #     break

        # test
        print()
        print("epoch test:", epoch)

        with torch.no_grad():
            loss = 0
            total = 0
            correct = 0
            for x, y in test_data:
                x = x[:, :, 0:3, :, :]
                x = x.type("torch.FloatTensor")
                if args.use_gpu:
                    x = x.cuda()
                    y = y.cuda()
                out = model.forward(x)
                loss += lossfunc(out, y).item()
                total += y.size(0)
                _, predicted = torch.max(out, 1)
                correct += (predicted == y).sum()

            print("total test samples:", total)
            print("loss:", loss / total)
            print("accuracy={}%".format(100 * correct / float(total)))
            model_path = os.path.join(args.save_path, "model.{}.pth".format(epoch))
            torch.save(model.state_dict(), model_path)
            print("saving model to {}".format(model_path))


if __name__ == "__main__":

    paser.add_argument("--image_path", type=str, required=True)
    paser.add_argument("--xml_path", type=str, required=True)
    paser.add_argument("--mode", type=str, default="train")
    paser.add_argument("--save_path", type=str, default="model")
    paser.add_argument("--model", type=str, default="")
    paser.add_argument("--batch_size", type=int, default=64)
    paser.add_argument("--worker", type=int, default=4)
    paser.add_argument("--epochs", type=int, default=10)
    paser.add_argument("--lr", type=float, default=0.0001)
    paser.add_argument("--use_gpu", type=bool, default=True)
    paser.add_argument("--num_class", type=int, default=2)
    args = paser.parse_args()

    xlist = os.listdir(args.xml_path)
    train_names, test_names = data_util.SplitData(args.image_path, "train", xlist, 0.5)

    dataset = TrafficDataSet(args.image_path, args.xml_path, train_names, 500, 400)
    train_loader = DataLoader(dataset, args.batch_size, num_workers=args.worker, shuffle=True)

    dataset = TrafficDataSet(args.image_path, args.xml_path, test_names, 500, 400)
    test_loader = DataLoader(dataset, args.batch_size, num_workers=args.worker, shuffle=False)

    # for x, y in test_loader:
    #     print(x.shape)

    print("Train data:", len(train_loader))
    print("Test data:", len(test_loader))
    print("Build Model......")
    model = Model(args.num_class)

    if args.use_gpu:
               model = model.cuda()

    print("Begin training......")
    train(train_loader, test_loader, model, args)
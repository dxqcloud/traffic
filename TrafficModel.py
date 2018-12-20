#encoding=utf-8
import os
import torch.nn as nn
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils.TrafficDataset import TrafficDataSet
from utils import data_util

paser = argparse.ArgumentParser()

input_w, input_h = (480, 320)

class Model(nn.Module):

    def __init__(self, num_class, base_model='resnet18'):
        super(Model, self).__init__()
        self.num_class = num_class
        self.base_model = base_model
        if self.base_model == 'alexnet':
            base = models.alexnet(True)
            base.features[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2,
                               bias=False)
            feature_size = 96771
        else:
            base = models.resnet18(False)
            base.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

            feature_size = 55296 + 3

        base = nn.Sequential(*list(base.children())[:-1])

        # resnet.fc = nn.LeakyReLU(0.1)

        for param in base.parameters():
            param.requires_grad = False

        self.base = base
        self.fc1 = nn.Linear(feature_size, 1024)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(1024, num_class)

        # print(self.resnet)

    def forward(self, input, type):
        feature = []
        # input = input[0]
        # print(input.size())
        for i in range(3):
            x = self.base(input[:,i,:,:,:])
            x = x.view(x.size(0), -1)
            feature.append(x)
        feature.append(type)
        feature = torch.cat(feature,1)
        # print(feature.size())
        out = self.fc1(feature)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def validation(test_data, model, args):
    with torch.no_grad():
        total = 0
        y_pred = []
        y_label = []
        for x, type, y in test_data:
            x = x.type("torch.FloatTensor")
            type = type.type("torch.FloatTensor")
            if args.use_gpu:
                x = x.cuda()
                type = type.cuda()
                y = y.cuda()
            out = model.forward(x, type)
            total += y.size(0)

            _, predicted = torch.max(out, 1)
            y_label.extend(y.cpu().data.numpy())
            y_pred.extend(predicted)
            print("accuracy={}%".format(100 * accuracy_score(y_label, y_pred)))

        print("total test samples:", total)
        print("accuracy={}%".format(100 * accuracy_score(y_label, y_pred)))
        print(confusion_matrix(y_label, y_pred))
        # print(classification_report(y_label, y_pred))


def train_model(train_data, test_data, model, args):
    lossfunc = nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # print("training samples:", len(train_data))
    for epoch in range(args.epochs):
        batch = 0
        for x, type, y in train_data:

            optimzer.zero_grad()
            x = x.type("torch.FloatTensor")
            type = type.type("torch.FloatTensor")
            if args.use_gpu:
                x = x.cuda()
                type = type.cuda()
                y = y.cuda()
            out = model.forward(x, type)
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
        validation(test_data, model, args)

        model_path = os.path.join(args.save_path, "model.{}.pth".format(epoch))
        torch.save(model.state_dict(), model_path)
        print("saving model to {}".format(model_path))


def predict_model(test_data, model, args):

    with torch.no_grad():
        y_pred = []
        for x, type in test_data:
            x = x.type("torch.FloatTensor")
            type = type.type("torch.FloatTensor")
            if args.use_gpu:
                x = x.cuda()
                type = type.cuda()
            out = model.forward(x, type)
            _, predicted = torch.max(out, 1)
            y_pred.extend(predicted)

    return y_pred


def train(args):

    xlist = os.listdir(args.xml_path)
    train_names, test_names = data_util.SplitData(args.image_path, "train", xlist, 0.5)

    transform = None
    if args.augmentation:

        transform = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1)
        ])

    dataset = TrafficDataSet(args.image_path, args.xml_path, train_names, input_w, input_h, transform=transform)
    train_loader = DataLoader(dataset, args.batch_size, num_workers=args.worker, shuffle=True)

    dataset = TrafficDataSet(args.image_path, args.xml_path, test_names, input_w, input_h)
    test_loader = DataLoader(dataset, args.batch_size, num_workers=args.worker, shuffle=False)
    #
    # for x, type, y in train_loader:
    #     print(x.shape, type.shape, y.shape)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    print("Train data:", len(train_loader))
    print("Test data:", len(test_loader))
    print("Build Model......")
    model = Model(args.num_class, args.model)
    if args.pretrain_model:
        model.load_state_dict(torch.load(args.pretrain_model))

    if args.use_gpu:
        model = model.cuda()

    print("Begin training......")
    train_model(train_loader, test_loader, model, args)

# for validation test
def test(args):

    xlist = os.listdir(args.xml_path)
    test_names = data_util.SplitData(args.image_path, "test", xlist, 0.5)

    dataset = TrafficDataSet(args.image_path, args.xml_path, test_names, input_w, input_h)
    test_loader = DataLoader(dataset, args.batch_size, num_workers=args.worker, shuffle=False)
    #
    # for x, type, y in train_loader:
    #     print(x.shape, type.shape, y.shape)

    print("Test data:", len(test_loader))
    print("Build Model......")
    model = Model(args.num_class, args.model)
    if args.pretrain_model:
        model.load_state_dict(torch.load(args.pretrain_model))

    if args.use_gpu:
        model = model.cuda()

    print("Begin test......")
    validation(test_loader, model, args)


def predict(args):

    test_names = os.listdir(args.image_path)

    dataset = TrafficDataSet(args.image_path, args.xml_path, test_names, input_w, input_h, mode="predict")
    test_loader = DataLoader(dataset, args.batch_size, num_workers=args.worker, shuffle=False)

    print("Test data:", len(test_loader))
    print("Build Model......")
    model = Model(args.num_class, args.model)

    if args.pretrain_model:
        model.load_state_dict(torch.load(args.pretrain_model))

    if args.use_gpu:
        model = model.cuda()

    print("Begin predict......")
    pred_result = predict_model(test_loader, model, args)

    with open(os.path.join(args.image_path, "result.txt"), "wb") as fd:
        for pred in pred_result:
            fd.writelines(str(pred))


if __name__ == "__main__":

    paser.add_argument("--image_path", type=str, required=True)
    paser.add_argument("--xml_path", type=str, required=True)
    paser.add_argument("--mode", type=str, default="train", choices=["train", "test", "predict"])
    paser.add_argument("--save_path", type=str, default="model")
    paser.add_argument("--model", type=str, default="alexnet")
    paser.add_argument("--pretrain_model", type=str, default="")
    paser.add_argument("--batch_size", type=int, default=64)
    paser.add_argument("--worker", type=int, default=4)
    paser.add_argument("--epochs", type=int, default=50)
    paser.add_argument("--lr", type=float, default=0.0001)
    paser.add_argument("--use_gpu", type=bool, default=True)
    paser.add_argument("--augmentation", type=bool, default=False)
    paser.add_argument("--num_class", type=int, default=2)
    args = paser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        predict(args)
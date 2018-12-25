#encoding=utf-8
import os

nlist = []

root = "train_data"

for dir in os.listdir(root):
    nlist.extend(os.listdir(os.path.join(root, dir)))

with open("{}.txt".format(root), "wb") as fd:
    for name in nlist:
        fd.write("{}\r\n".format(name))
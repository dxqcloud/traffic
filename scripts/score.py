#encoding=utf-8

import os
import sys

def stats(result):

    print(result)
    stats1 = {"1208": 0, "1625": 0, "1345": 0}
    stats0 = {"1208": 0, "1625": 0, "1345": 0}
    num = {"1208": 0, "1625": 0, "1345": 0}

    with open(result) as fd:
        lines = fd.readlines()
        for line in lines:
            pred, name = line.split("#")
            items = name.strip().split("_")
            type = items[0]
            num[type] += 1

            if pred == "1":
                stats1[type] += 1
            if pred == "0":
                stats0[type] += 1

    return num, stats0, stats1


def cal_score(result_file, ans_dir):

    anslist = os.listdir(ans_dir)
    ans_dict = {}
    for ans in anslist:
        items = ans.strip().split("_")
        # 违法行为_场景编号_号牌号码_图片序号.jpg
        ans_dict[items[-1]] = items[0]

    stats11 = {"1208":0, "1625":0, "1345":0}
    stats10 = {"1208":0, "1625":0, "1345":0}
    stats01 = {"1208":0, "1625":0, "1345":0}
    num = {"1208":0, "1625":0, "1345":0}
    with open(result_file) as fd:
        lines = fd.readlines()
        for line in lines:
            pred, name = line.split("#")
            items = name.strip().split("_")
            label = ans_dict[items[-1]]
            type = items[0]

            num[type] += 1

            if label == "1":
                if pred == "1":
                    stats11[type] += 1
                if pred == "0":
                    stats10[type] += 1
            if label == "0":
                if pred == "1":
                    stats01[type] += 1

    precision = []
    recall = []
    score = []
    for type in ["1208", "1625", "1345"]:
        p = stats11[type] / (stats11[type] + stats01[type])
        r = stats11[type] / (stats11[type] + stats10[type])
        precision.append(p)
        recall.append(r)
        s = p * 0.3 + r * 0.7
        score.append(s)

    result = score[0] * 0.3 + score[1] * 0.3 + score[2] * 0.4

    return num, result, precision, recall

def get_score(pred_label, anslist):


    stats11 = {"1208":0, "1625":0, "1345":0}
    stats10 = {"1208":0, "1625":0, "1345":0}
    stats01 = {"1208":0, "1625":0, "1345":0}
    num = {"1208":0, "1625":0, "1345":0}

    for pred, name in zip(pred_label, anslist):

        items = name.strip().split("_")
        label = items[0]
        type = items[1]

        num[type] += 1

        if label == "1":
            if pred == 1:
                stats11[type] += 1
            if pred == 0:
                stats10[type] += 1
        if label == "0":
            if pred == 1:
                stats01[type] += 1

    precision = []
    recall = []
    score = []
    for type in ["1208", "1625", "1345"]:
        p = stats11[type] / (stats11[type] + stats01[type])
        r = stats11[type] / (stats11[type] + stats10[type])
        precision.append(p)
        recall.append(r)
        s = p * 0.3 + r * 0.7
        score.append(s)

    result = score[0] * 0.3 + score[1] * 0.3 + score[2] * 0.4
    return num, result, precision, recall


if __name__ == "__main__":

    num, result, p, r = cal_score(sys.argv[1], sys.argv[2])
    print(num)
    print("score:", result)
    print("precision:", p)
    print("recall:", r)
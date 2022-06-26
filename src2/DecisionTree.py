import math
import numpy as np


def shannonEntropy(p):
    q = 1 - p
    ans = 0
    if p > 0:
        ans -= p * math.log2(p)
    if q > 0:
        ans -= q * math.log2(q)
    return ans


def calculateNewEntropy(attributes, labels):
    data = np.column_stack((attributes, labels))
    cnt = {}
    trueCnt = {}
    for d in data:
        if d[0] not in cnt.keys():
            cnt[d[0]] = 1
            trueCnt[d[0]] = 0
        else:
            cnt[d[0]] += 1
        if d[1]:
            trueCnt[d[0]] += 1
    entropy = 0
    for key in cnt:
        entropy += shannonEntropy(trueCnt[key] / cnt[key]) * (
            cnt[key] / np.size(labels)
        )
    return entropy


class DecisionTree:
    def __init__(self):
        self.label = -1
        self.decisionAttribute = -1
        self.kids = {}
        self.AttrAvailability = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

    def fit(self, train_features, train_labels):
        tempSum = np.sum(train_labels)
        if tempSum == 0:
            self.label = 0
            return
        if tempSum == np.size(train_labels):
            self.label = 1
            return
        if np.sum(self.AttrAvailability) == 0:
            if tempSum < np.size(train_labels) / 2:
                self.label = 0
                return
            else:
                self.label = 1
                return
        flag = True
        for i in range(1, np.size(train_labels)):
            if not np.array_equal(train_features[0], train_features[i]):
                flag = False
                break
        if flag:
            if tempSum < np.size(train_labels) / 2:
                self.label = 0
                return
            else:
                self.label = 1
                return
        if tempSum < np.size(train_labels) / 2:
            self.label = 0
        else:
            self.label = 1
        min = 114  # shannon entropy<=logN
        minIdx = -1
        for i in range(9):
            if self.AttrAvailability[i]:
                entropy = calculateNewEntropy(train_features[:, i], train_labels)
                if entropy < min:
                    min = entropy
                    minIdx = i
        self.decisionAttribute = minIdx
        attributes = train_features[:, minIdx]
        for attr in attributes:
            if attr not in self.kids.keys():
                self.kids[attr] = DecisionTree()
                self.kids[attr].AttrAvailability[minIdx] = 0
        for key in self.kids.keys():
            newFeature = []
            newLabel = []
            for i in range(np.size(train_labels)):
                if train_features[i, minIdx] == key:
                    newFeature.append(train_features[i])
                    newLabel.append(train_labels[i])
            self.kids[key].fit(np.array(newFeature), np.array(newLabel))

    def predict(self, test_features):
        ans = []
        for feature in test_features:
            node = self
            while node.decisionAttribute != -1:
                if feature[node.decisionAttribute] in node.kids:
                    # print(node.decisionAttribute, end="")
                    node = node.kids[feature[node.decisionAttribute]]
                else:
                    break
            # print()
            ans.append(node.label)
        return np.array(ans)


# treenode: [attr, feat[attr] == 1, feat[attr] == 2, feat[attr] == 3]

import torch
import numpy as np
from matplotlib import pyplot as plt
import math


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


class MLP:
    def __init__(self):
        # layer size = [10, 8, 8, 4]
        # 初始化所需参数
        self.w1 = np.random.uniform(
            low=-math.sqrt(6 / 10 + 10), high=math.sqrt(6 / 10 + 10), size=(10, 10)
        )
        self.w2 = np.random.uniform(
            low=-math.sqrt(6 / 8 + 10), high=math.sqrt(6 / 8 + 10), size=(8, 10)
        )
        self.w3 = np.random.uniform(
            low=-math.sqrt(6 / 8 + 8), high=math.sqrt(6 / 8 + 8), size=(8, 8)
        )
        self.w4 = np.random.uniform(
            low=-math.sqrt(6 / 4 + 8), high=math.sqrt(6 / 4 + 8), size=(4, 8)
        )
        self.b1 = np.zeros((10, 1))
        self.b2 = np.zeros((8, 1))
        self.b3 = np.zeros((8, 1))
        self.b4 = np.zeros((4, 1))

    def forward(self, x):
        h1 = np.tanh(self.w1 @ x + self.b1)
        h2 = np.tanh(self.w2 @ h1 + self.b2)
        h3 = np.tanh(self.w3 @ h2 + self.b3)
        h4 = softmax(self.w4 @ h3 + self.b4)
        return [h1, h2, h3, h4]

    def backward(self, y, label_, x):  # 自行确定参数表
        # 反向传播
        label = label_.reshape((4, 1))
        db4 = y[3] - label
        db3 = (self.w4.T @ db4) * (1 - y[2] ** 2)
        db2 = (self.w3.T @ db3) * (1 - y[1] ** 2)
        db1 = (self.w2.T @ db2) * (1 - y[0] ** 2)
        dw4 = db4 @ (y[2].T)
        dw3 = db3 @ (y[1].T)
        dw2 = db2 @ (y[0].T)
        dw1 = db1 @ (x.reshape((1, 10)))
        return [dw1, dw2, dw3, dw4, db1, db2, db3, db4]


def train(mlp: MLP, epochs, lr, inputs, labels):
    losses = []
    for k in range(epochs):
        dw1 = np.zeros((10, 10))
        dw2 = np.zeros((8, 10))
        dw3 = np.zeros((8, 8))
        dw4 = np.zeros((4, 8))
        db1 = np.zeros((10, 1))
        db2 = np.zeros((8, 1))
        db3 = np.zeros((8, 1))
        db4 = np.zeros((4, 1))
        loss = 0
        for i in range(100):
            hs = mlp.forward(inputs[i].reshape((10, 1)))
            loss -= math.log(np.sum(labels[i] * (hs[3]).reshape(4))) / 100
            grad = mlp.backward(hs, labels[i], inputs[i])
            dw1 += grad[0] / 100
            dw2 += grad[1] / 100
            dw3 += grad[2] / 100
            dw4 += grad[3] / 100
            db1 += grad[4] / 100
            db2 += grad[5] / 100
            db3 += grad[6] / 100
            db4 += grad[7] / 100
        print("{}: {}".format(k, loss))
        losses.append(loss)
        mlp.w1 -= lr * dw1
        mlp.w2 -= lr * dw2
        mlp.w3 -= lr * dw3
        mlp.w4 -= lr * dw4
        mlp.b1 -= lr * db1
        mlp.b2 -= lr * db2
        mlp.b3 -= lr * db3
        mlp.b4 -= lr * db4
    return losses


if __name__ == "__main__":
    # 设置随机种子,保证结果的可复现性
    np.random.seed(1)
    mlp = MLP()
    # 生成数据
    inputs = np.random.randn(100, 10)

    # 生成one-hot标签
    labels = np.eye(4)[np.random.randint(0, 4, size=(1, 100))].reshape(100, 4)

    # 训练
    loss = []
    loss += train(mlp, 1000, 0.5, inputs, labels)
    loss += train(mlp, 1000, 0.4, inputs, labels)
    loss += train(mlp, 1000, 0.2, inputs, labels)
    loss += train(mlp, 1500, 0.1, inputs, labels)
    plt.plot(loss)
    plt.show()

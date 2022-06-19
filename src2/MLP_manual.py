import torch
import numpy as np
from matplotlib import pyplot as plt

class MLP:
    def __init__(self):
        # layer size = [10, 8, 8, 4]
        # 初始化所需参数   
        pass

    def forward(self, x):
        # 前向传播
        pass

    def backward(self): # 自行确定参数表
        # 反向传播
        pass

def train(mlp: MLP, epochs, lr, inputs, labels):
    '''
        mlp: 传入实例化的MLP模型
        epochs: 训练轮数
        lr: 学习率
        inputs: 生成的随机数据
        labels: 生成的one-hot标签
    '''
    
    pass


if __name__ == '__main__':
    # 设置随机种子,保证结果的可复现性
    np.random.seed(1)
    mlp = MLP()
    # 生成数据
    inputs = np.random.randn(100, 10)

    # 生成one-hot标签
    labels = np.eye(4)[np.random.randint(0, 4, size=(1, 100))].reshape(100, 4)

    # 训练
    # train(mlp, epochs, lr, inputs, labels)
    
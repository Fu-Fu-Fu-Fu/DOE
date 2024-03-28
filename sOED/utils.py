import numpy as np
import torch
import torch.nn as nn
# import warnings
# warnings.filterwarnings("ignore")

# logpdf of independent normal distribution.
# x is of size (n_sample, n_param).
# loc and scale are int or numpy.ndarray of size n_param.
# output is of size n_sample.
def norm_logpdf(x, loc=0, scale=1):#正态分布的对数概率密度函数，x是随机变量，loc为均值，scale为标准差
    logpdf = (-np.log(np.sqrt(2 * np.pi) * scale) 
              - (x - loc) ** 2 / 2 / scale ** 2)
    return logpdf.sum(axis=-1)

# pdf of independent normal distribution.
def norm_pdf(x, loc=0, scale=1):
    return np.exp(norm_logpdf(x, loc, scale))#取指数

# logpdf of uniform distribution.
def uniform_logpdf(x, low=0, high=1):
    return np.log(uniform_pdf(x, low, high))

# pdf of uniform distribution.
def uniform_pdf(x, low=0, high=1):#均匀分布
    pdf = ((x >= low) * (x <= high)) / (high - low)
    return pdf.prod(axis=1)

# Construct neural network
class Net(nn.Module):#构建神经网络
    def __init__(self, dimns, activate, bounds):#self(指向新创建对象的引用)、dimns（神经网络各层的维度）、activate（激活函数）、bounds（神经元输出的上下限）
        super().__init__()
        layers = []#创建空列表，存储要添加的神经网络层
        for i in range(len(dimns) - 1):
            layers.append(nn.Linear(dimns[i], dimns[i + 1]))
            if i < len(dimns) - 2:
                layers.append(activate)
        self.net = nn.Sequential(*layers)
        self.bounds = torch.from_numpy(bounds)
        self.has_inf = torch.isinf(self.bounds).sum()

    def forward(self, x):#定义向前传播函数
        x = self.net(x)
        if self.has_inf:
            x = torch.maximum(x, self.bounds[:, 0])
            x = torch.minimum(x, self.bounds[:, 1])
        else:
            x = (torch.sigmoid(x) * (self.bounds[:, 1] - 
                                     self.bounds[:, 0]) + self.bounds[:, 0])
        return x

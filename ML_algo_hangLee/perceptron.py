#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/6/11 0:12
# @Author  : Leslee
import random
import time
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
###
# 参考github博主的实现
###
# todoList:建立一个文本文件，特征抽取和训练预测。

class Perceptron(object):

    def __init__(self):
        self.learning_rate = 0.00001
        self.max_iter = 5000
# 这里给出原始形式的训练
    def sign(self,v):
        if v >= 0:
            return 1
        else:
            return -1
    def train_origin(self,features,labels):
        self.w = [0.0] * len(features[0])
        self.b = [0.0] * len(features[0])
        current_count = 0
        iterNum = 0
        while iterNum < self.max_iter:
            index = random.randint(0, len(labels) - 1)  # 随机采样
            x = list(features[index])
            y = 2 * labels[index] - 1
            if(y*self.sign(sum([self.w[j]*x[j]+self.b[j] for j in range(len(self.w))]))<=0):
                for i in range(len(self.w)):
                    self.w[i] += self.learning_rate*y*x[i]
                    self.b[i] += self.learning_rate*y
            else:
                current_count +=1
                if current_count > self.max_iter:
                    break
################################
# 这里给出感知机的对偶实现
    # 计算Gram矩阵
    def cal_gmatrix(self,features):
        fea_len = len(features)
        self.G_matrix = np.zeros((fea_len,fea_len))
        for i in range(fea_len):
            for j in range(fea_len):
                self.G_matrix[i][j] = np.sum(features[i]*features[j])

    def judge(self,features,labels,index):
        tmp = self.b
        fea_len = len(features)
        for i in range(fea_len):
            tmp += self.alpha[i] * labels[i] * self.G_matrix[index][i]
    def train_dual(self,features,labels):
        fea_len = len(features)
        self.w = [0.0] * fea_len
        self.b = 0
        self.cal_gmatrix(features)
        i = 0
        while i<fea_len:
            index = random.randint(0, len(labels) - 1)  # 随机采样
            x = list(features[index])
            y = 2 * labels[index] - 1
            if(y*self.sign(sum([self.w[j]*x[j]+self.b for j in range(len(self.w))]))<=0):
                pass
################################

# 这里采用的一个非常巧妙地方法进行训练，参考李航的收敛部分的证明
    def train(self,features,labels):
        self.w = [0.0] * (len(features[0])+1)

        current_count = 0
        iterNum = 0
        while iterNum < self.max_iter:
            index = random.randint(0,len(labels)-1) #随机采样

            x = list(features[index])
            x.append(1.0) # 算法收敛性一章，对输入加入常数1，转化为矩阵运算。
            y = 2 * labels[index] -1 # norm 输出标签为[-1,1]
            wx = sum([self.w[j]*x[j] for j in range(len(self.w))])# 这里实际上是(w',b)' *(x',1.0)
            if wx * y >0:
                current_count += 1
                if current_count > self.max_iter:
                    break
                continue

            for i in range(len(self.w)):
                self.w[i] += self.learning_rate * (y*x[i]) # 根据算法收敛性推倒
    def predict_(self,x):# 给定x，通过学习得到的矩阵进行预测，当计算结果大于0的时候，说明是正确分类点，
        wx = sum([self.w[j]*x[j] for j in range(len(self.w))])
        return int(wx>0)

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels

if __name__ == '__main__':
    time1 = time.time()
    raw_data = pd.read_csv('../data/train_binary.csv',header=0)
    data = raw_data.values
    imgs = data[0::,1::]
    labels = data[::,0]
    train_features,test_features,train_labels,test_labels = train_test_split(
        imgs,labels,test_size=0.3,random_state=23333
    )
    time_2 = time.time()
    print('read data cost ', time_2 - time1, ' second', '\n')
    p = Perceptron()
    p.train_origin(train_features,train_labels)

    test_predict = p.predict(test_features)
    score = accuracy_score(test_labels, test_predict)
    print("The accruacy socre is ", score)
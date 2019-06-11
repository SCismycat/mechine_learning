#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/6/11 0:12
# @Author  : Leslee
import random
class Perceptron(object):

    def __init__(self):
        self.learning_rate = 0.00001
        self.max_iter = 5000

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
                self.w[i] += self.learning_rate * (y*x[i])
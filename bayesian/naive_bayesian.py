#coding: utf-8
#date: 2016-07-10
#mail: artorius.mailbox@qq.com
#author: xinwangzhong -version 0.1

from numpy import *

def trainNB0(trainMatrix,trainCatergory):
	#适用于二分类问题，其中一类的标签为1
    #return
    #p0Vect：标签为0的样本中，出现某个特征对应的概率
    #p1Vect：标签为1的样本中，出现某个特征对应的概率
    #pAbusive：标签为1的样本出现的概率
    numTrainDoc = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCatergory)/float(numTrainDoc)
    #防止多个概率的成绩当中的一个为0
    #p0Num: 在训练样本标签为0的数据中，所有特征的对应value值之和，为矩阵
    #p1Num: 在训练样本标签为1的数据中，所有特征的对应value值之和，为矩阵
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    #p0Denom：在训练样本标签为0的数据中，所有特征的value值之和，为标量
    #p1Denom：在训练样本标签为1的数据中，所有特征的value值之和，为标量
    #为什么初始化为2？？
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDoc):
        if trainCatergory[i] == 1:
            p1Num +=trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num +=trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #出于精度的考虑，否则很可能到限归零，change to log()
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
    
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #element-wise mult,只算分子的log值，因为只需比较大小，所以正负无关
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

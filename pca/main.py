#coding: utf-8
#date: 2016-07-01
#mail: artorius.mailbox@qq.com
#author: xinwangzhong -version 0.1

import sys
from numpy import *
import pylab
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):    
    # 去除平均值
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵
    covMat = cov(meanRemoved, rowvar=0)
    # 计算特征值eigVals, 特征向量eigVects
    eigVals, eigVects = linalg.eig(mat(covMat))
    print eigVects
    num_eigvals = len(eigVals)
    # 从大到小排序
    eigValInd = argsort(-eigVals)
    # 提取前topNfeat特征向量
    eigValInd = eigValInd[:topNfeat]
    # reorganize eigVects
    redEigVects = eigVects[:,eigValInd]
    # print meanRemoved*redEigVects*redEigVects.T, meanRemoved
    # sys.exit()
    # 映射到新基后的数据（rotation）
    print redEigVects
    lowDDataMat = meanRemoved * redEigVects
    # 还原后的数据原始数据：
    # meanRemoved * redEigVects * redEigVects.T + meanVals
    # 其中redEigVects * redEigVects.T为单位矩阵
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def plotBestFit(dataSet1,dataSet2):
    dataArr1 = array(dataSet1)
    dataArr2 = array(dataSet2)
    n = shape(dataArr1)[0] 
    n1=shape(dataArr2)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    xcord3=[];ycord3=[]
    j=0
    for i in range(n):
            xcord1.append(dataArr1[i,0]); ycord1.append(dataArr1[i,1])
            xcord2.append(dataArr2[i,0]); ycord2.append(dataArr2[i,1])
                   
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='blue')
    
    plt.xlabel('X1'); 
    plt.ylabel('X2');
    plt.show()    

if __name__=='__main__':
     # mata = loadDataSet('testSet.txt')
     x = random.normal(10,.5,1000)  
     y = random.normal(1,1,1000)  
     a = x*cos(pi/4) + y*sin(pi/4)  
     b = -x*sin(pi/4) + y*cos(pi/4)  
     pylab.plot(a,b,'.')  
     pylab.xlabel('x')  
     pylab.ylabel('y')  
     pylab.title('raw dataset')  
     data = zeros((1000,2))  
     data[:,0] = a  
     data[:,1] = b
     topNfeat = 2
     lowDDataMat,reconMat = pca(data, topNfeat)
     if (len(data[1])!=topNfeat):
	     lowDDataMat = map(lambda x:[x[0], [0]*(len(data[1])-topNfeat)], list(lowDDataMat))
     plotBestFit(lowDDataMat, reconMat)

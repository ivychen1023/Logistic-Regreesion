#coding:utf-8 用于解决注释中出现中文报错的问题
from numpy import *
def loadDataSet():
	dataMat = [] #代表参数x[i]
	labelMat = [] #代表y
	fr = open('RegressionData.txt')
	for line in fr.readlines(): # 一行一行的读取数据
		lineArr = line.strip().split()  #读的是txt文件中的每一行 返回的是一个list 如[x1,x2,y]=['-0.017612', '14.053064', '0']
		#print(lineArr)
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) #分别取type是list的lineArr中代表是特征x1=lineArr[0]  x2=lineArr[1]
		labelMat.append(int(lineArr[2])) #取的是type是list的lineArr中代表是标签的y = lineArr[2]
	return dataMat,labelMat	 #两组数据的type都是list

#定义Sigmoid函数
def sigmoid(inX):
	return 1.0/(1+exp(-inX))

#定义求解最佳回归系数 

def gradAscent(dataMatIn,classLabels):
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose() #因为classLabels（即为第一个函数中的labelMat）是一个横向的list 所以得转置为纵向的一维矩阵
	m,n = shape(dataMatrix)
	alpha = 0.00100
	maxCycles = 500 #迭代次数共500次 
	weights = ones((n,1)) #初始化参数weights
	print(weights) 
	for k in range(maxCycles): #迭代500次
		h = sigmoid(dataMatrix * weights)
		error = labelMat - h
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights

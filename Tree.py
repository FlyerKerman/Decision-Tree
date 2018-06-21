# -*- coding: utf-8 -*-
import numpy as np
from math import log
from collections import Counter


def indFinder(dataSet,feature,value):
	m,n = np.shape(np.mat(dataSet))
	ind0 = []
	ind1 = []
	for i in range(m):
		if float(dataSet[i,feature])<value:
			ind0.append(i)
		else:
			ind1.append(i)
	return ind0,ind1

def binSplitDataSet(dataMat, labelMat,feature, value):
	ind0,ind1 = indFinder(dataMat,feature,value)
	print("Split函数决定这么分：",ind0,ind1)
	mat0 = dataMat[np.mat(ind0),:]
	mat1 = dataMat[np.mat(ind1),:]
	labels = np.mat(labelMat)
	label0 = labels[0,np.mat(ind0)]
	label1 = labels[0,np.mat(ind1)]
	print("分割结果:",ind0,ind1)
	return mat0[0],mat1[0],ind0,ind1,label0.tolist()[0],label1.tolist()[0]

def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

def chooseBestSplit(dataSet,labelMat,currentDepth,LeafType=regLeaf):
	dataSet1 = np.mat(dataSet)
	val = Counter(labelMat).most_common(1)[0][0]#找到出现次数最多的标签
	if(currentDepth>5):#如果深度大于5则停止分割
		return None,val
	if(len(set(labelMat))==1):#仅靠信息增益来判断似乎不能发现这种待分样本标签全部一致一样的情况
		return None,labelMat[0]
	a,b = np.shape(dataSet1)
	dataMat1 = np.zeros((a,b-1))#从字符串格式的dataMat获取float格式的数据矩阵
	for i in range(a):
		for j in range(b-1):
			dataMat1[i,j] = float(dataSet1[i,j])
	if len(set(dataSet1[:,-1].T.tolist()[0])) == 1:
		return None,dataSet1[0,-1]
	m,n = np.shape(dataMat1)
	S = calcEntropy(labelMat,range(m))
	bestS = np.inf; bestIndex = 0; bestValue = 0
	####################排序并取相邻两项均值作为分界###########################
	for featIndex in range(n):
		print(featIndex)
		sortedFeat = sorted(set(dataMat1[:,featIndex].T.tolist()))
		splitVals = []
		for i in range(len(sortedFeat)-1):#分界值并非直接从该属性的值中选取，而是在去重并排列后取各相邻两值的均值
			splitVals.append(0.5*float(sortedFeat[i]+sortedFeat[i+1]))
		for splitVal in splitVals: #set(dataSet[:,featIndex]):#遍历所有分界值，选择信息增益最高的分割
			mat0,mat1,ind0,ind1,label0,label1 = binSplitDataSet(dataSet1, labelMat,featIndex, splitVal)
			Num = len(mat0) + len(mat1)
			newS = (len(mat0)/Num)*calcEntropy(labelMat,ind0) + (len(mat1)/Num)*calcEntropy(labelMat,ind1)
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	if (S - bestS) < 0.5: 
		return None, dataSet[0][-1] 
	mat0,mat1,ind0,ind1,label0,label1 = binSplitDataSet(dataSet1, labelMat,bestIndex, bestValue)
	#if(bestS>=S):
		#return None,val
	return bestIndex,bestValue
def calcEntropy(dataSet,inds):#用于计算某组样本的信息熵。通过当前节点标签集和索引数组进行读取和计算
	#print(inds)
	num = len(inds)
	labelCounts={}
	if num==0:return 0
	if(type(dataSet)==str):
		return 0
	for i in inds:
		if i==0:
			print("dataSet的shape：",np.shape(dataSet))
		if len(np.shape(dataSet))>=2:
			dataSet = dataSet[0,:]
		currentLabel = dataSet[i]
		print("currentLabel:",currentLabel)
		print("i的值：",i)
		if currentLabel not in list(labelCounts.keys()):
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/num
		shannonEnt -= prob * log(prob,2)
	return shannonEnt

def loadDataSet(fileName):#在连续与离散特征都有的情况下还需要另一个函数来判断特征使离散的还是连续的
	dataMat = []
	labelMat = []
	dataSet = []
	fr = open(fileName)
	fr.readline()
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		dataSet.append(curLine)
		labelMat.append(curLine[-1])
		fltLine = map(float,curLine[:-1])
		dataMat.append(list(fltLine))
	fr.close()
	return dataMat,labelMat,dataSet


def createTree(dataSet,labelMat,fatherDepth, leafType=regLeaf):
	currentDepth = fatherDepth + 1;#当前深度等于父节点深度+1
	feat, val = chooseBestSplit(dataSet,labelMat,currentDepth, regLeaf)
	if feat == None: return val 
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	print("val的值:",val)
	lSet, rSet,what,ever,label0,label1 = binSplitDataSet(np.mat(dataSet),labelMat,feat, val)
	retTree['left'] = createTree(lSet,label0,currentDepth, regLeaf)
	retTree['right'] = createTree(rSet,label1,currentDepth, regLeaf)
	if(retTree['left']==retTree['right']):
		return retTree['left']
	return retTree  
if __name__ == '__main__':

	dataMat,labelMat,dataSet = loadDataSet('IrisDataset.txt')
	tree = createTree(dataSet,labelMat,1,regLeaf)
	print(tree)

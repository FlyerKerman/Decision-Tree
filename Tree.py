# -*- coding: utf-8 -*-
import numpy as np
from math import log
from collections import Counter

#global times
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
	#print(np.nonzero(dataMat[:,feature] > value)[0])
	ind0,ind1 = indFinder(dataMat,feature,value)
	print("Split函数决定这么分：",ind0,ind1)
	mat0 = dataMat[np.mat(ind0),:]
	mat1 = dataMat[np.mat(ind1),:]
	print("labelMat的状况:",labelMat,type(labelMat))
	labels = np.mat(labelMat)
	label0 = labels[0,np.mat(ind0)]
	label1 = labels[0,np.mat(ind1)]
	#print("mat0的shape\n")
	#print(np.shape(mat0))
	#print("mat1的shape\n")
	#print(np.shape(mat1))
	#mat0 = dataMat[np.nonzero(dataMat[:,feature] - value)[0],:]
	#mat1 = dataMat[np.nonzero(dataMat[:,feature] - value)[0],:]
	#ind0 = np.nonzero(dataMat[:,feature] > value)[0]
	#ind1 = np.nonzero(dataMat[:,feature] <= value)[0]
	print("分类结果:",ind0,ind1)
	return mat0[0],mat1[0],ind0,ind1,label0.tolist()[0],label1.tolist()[0]

def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

#def chooseBestSplit(dataSet,dataMat, leafType=regLeaf, errType=regErr, ops=(1,4)):
def chooseBestSplit(dataSet,labelMat,currentDepth,LeafType=regLeaf):
	dataSet1 = np.mat(dataSet)
	val = Counter(labelMat).most_common(1)[0][0]#找到出现次数最多的标签
	if(currentDepth>=5):
		return None,val
	if(len(set(labelMat))==1):#仅靠信息增益来判断似乎不能发现这种待分样本标签全部一致一样的情况
		return None,labelMat[0]
	#print("dataSet1的shape\n")
	#print(np.shape(dataSet1))
	a,b = np.shape(dataSet1)
	dataMat1 = np.zeros((a,b-1))
	for i in range(a):
		for j in range(b-1):
			dataMat1[i,j] = float(dataSet1[i,j])
	#dataMat1 = dataSet1[:,:-1]
	#if all the target variables are the same value: quit and return value
	if len(set(dataSet1[:,-1].T.tolist()[0])) == 1: #ex`it cond 1
		return None,dataSet1[0,-1]#, leafType(dataSet)
	m,n = np.shape(dataMat1)
	print("此时的labelMat：",labelMat)
	print("labelMat的类型：",type(labelMat))
	#print("labelMat[0][:]的尺寸:",len(labelMat[0][:]))
	S = calcEntropy(labelMat,range(m))
	bestS = np.inf; bestIndex = 0; bestValue = 0
	####################排序并取相邻两项均值作为分界###########################
	for featIndex in range(n):
		print(featIndex)
		#print(dataMat1)
		#print(dataMat1[:,featIndex].T.tolist())
		sortedFeat = sorted(set(dataMat1[:,featIndex].T.tolist()))
		#print(sortedFeat)
		#sortedFeat = set(sortedFeat)
		splitVals = []
		for i in range(len(sortedFeat)-1):
			splitVals.append(0.5*float(sortedFeat[i]+sortedFeat[i+1]))
		for splitVal in splitVals: #set(dataSet[:,featIndex]):
			mat0,mat1,ind0,ind1,label0,label1 = binSplitDataSet(dataSet1, labelMat,featIndex, splitVal)
			#print(ind0)
			#print(ind1)
			Num = len(mat0) + len(mat1)
			#if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
			#if (len(ind0) < tolN) or (len(ind1) < tolN): continue
			#newS = errType(mat0) + errType(mat1)
			newS = (len(mat0)/Num)*calcEntropy(labelMat,ind0) + (len(mat1)/Num)*calcEntropy(labelMat,ind1)
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	#if (S - bestS) < tolS: 
		#return None, dataSet[0][-1] 
	mat0,mat1,ind0,ind1,label0,label1 = binSplitDataSet(dataSet1, labelMat,bestIndex, bestValue)
	#if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
	#if (len(ind0) < tolN) or (len(ind1) < tolN):  #exit cond 3
		#targ = Counter(dataSet1[:,-1].T.tolist()[0]).most_common(1)[0]
		#return None, targ
	if(bestS>=S):
		return None,val
	return bestIndex,bestValue#returns the best feature to split on
							#and the value used for that split

def calcEntropy(dataSet,inds):
	#print(inds)
	num = len(inds)
	labelCounts={}
	if num==0:return 0
	if(type(dataSet)==str):
		return 0
	for i in inds:
		#print("dataSet的shape：",np.shape(dataSet))
		#print("currentLabel应该是:",dataSet[0][2])
		if i==0:
			#print(dataSet)
			print("dataSet的shape：",np.shape(dataSet))
		if len(np.shape(dataSet))>=2:
			dataSet = dataSet[0,:]
		#print(type(i))
		currentLabel = dataSet[i]
		print("currentLabel:",currentLabel)
		print("len",len(currentLabel))
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

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		pass

	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass

	return False

def createTree(dataSet,labelMat,fatherDepth, leafType=regLeaf):#assume dataSet is NumPy Mat so we can array filtering
	#global times
	#times = times + 1
	currentDepth = fatherDepth + 1;
	feat, val = chooseBestSplit(dataSet,labelMat,currentDepth, regLeaf)#choose the best split
	if feat == None: return val #if the splitting hit a stop condition return val
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	print("val的值:",val)
	lSet, rSet,what,ever,label0,label1 = binSplitDataSet(np.mat(dataSet),labelMat,feat, val)
	#print("lSet的shape\n")
	#print(np.shape(lSet))
	#print("rSet的shape\n")
	#print(np.shape(rSet))
	retTree['left'] = createTree(lSet,label0,currentDepth, regLeaf)
	retTree['right'] = createTree(rSet,label1,currentDepth, regLeaf)
	if(retTree['left']==retTree['right']):
		return retTree['left']
	#print(retTree['spInd'])
	return retTree  
if __name__ == '__main__':
	#global times
	#times = 0;
	dataMat,labelMat,dataSet = loadDataSet('IrisDataset.txt')
	tree = createTree(dataSet,labelMat,1,regLeaf)
	print(tree)
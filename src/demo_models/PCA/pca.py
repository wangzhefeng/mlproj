# -*- coding: utf-8 -*-
# @Date    : 2017-09-03 22:28:16
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

# PCA

import numpy as np
from numpy import linalg

def loadDataSet(fileName, delim = '\t'):
	fr = open(fileName)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	dataArr = list(map(float, line) for line in stringArr)
	return np.mat(dataArr)

def pca(dataMat, topNfeat = 9999999):
	meanVals = np.mean(dataMat, axis = 0)
	meanRemoved = dataMat - meanVals

	covMat = np.cov(meanRemoved, rowvar = 0)
	eigVals, eigVects = linalg.eig(mat(covMat))
	eigValInd = argsort(eigVals)
	eigValInd = eigValInd[:-(topNfeat + 1):-1]
	redEigVects = eigvects[:, eigValInd]
	lowDDataMat = meanRemoved * redEigVects
	reconMat = (lowDDataMat * redEigVects.T) + meanVals
	return lowDDataMat, reconMat

dataMat = loadDataSet('/home/wangzhefeng/project/python/testSet.txt')
print(list(dataMat))
lowDMat, reconMat = pca(dataMat = dataMat, topNfeat = 1)
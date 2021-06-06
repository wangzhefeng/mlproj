# -*- coding: utf-8 -*-

'这是一个文档注释'

__author__ = 'Tinker Wang'

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import random
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

##############################################################################
#-------------------- 读取titanic训练和测试数据维一个DataFrame
titanic_df = pd.read_csv("E:\\GitHub\\data\\titanic\\train.csv")
test_df = pd.read_csv("E:\\GitHub\\data\\titanic\\test.csv")
print(titanic_df.head())
print("---------------------------------------------")
print(test_df.head())

print(titanic_df.info())
print("---------------------------------------------")
print(test_df.info())


##############################################################################
#-------------------- 去除在之后分析和预测中无用的变量
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)
test_df = test_df.drop(['Name', 'Ticket'], axis = 1)
print("--------------------------------------")
print(titanic_df.head())
print(test_df.head())


##############################################################################
#-------------------- 特征工程
print(titanic_df.info())
# Survived    891 non-null int64
# Pclass      891 non-null int64
# Sex         891 non-null object
# Age         714 non-null float64       (有缺失值 77个)
# SibSp       891 non-null int64
# Parch       891 non-null int64
# Fare        891 non-null float64
# Cabin       204 non-null object        (有缺失值 687个)
# Embarked    889 non-null object        (有缺失值 2个)
print(test_df.info())
# PassengerId    418 non-null int64
# Pclass         418 non-null int64
# Sex            418 non-null object
# Age            332 non-null float64    (有缺失值 86个)
# SibSp          418 non-null int64
# Parch          418 non-null int64
# Fare           417 non-null float64    (有缺失值 1个)
# Cabin          91 non-null object      (有缺失值 327个)
# Embarked       418 non-null object


## 变量Embarked只有在titanic_df中有缺失值,且只有两个缺失值,取该变量的众数填充
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
sns.factorplot('Embarked', 'Survived', data = titanic_df, size = 4, aspect = 3)
## 变量Age在titanic_df和test_df中都有少量的缺失值


## 变量Cabin在titanic_df和test_df中都有大量的缺失值



## 变量Fare只在test_df中有缺失值,且只有一个缺失值,取该变量的众数填充
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)





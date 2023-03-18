# -*- coding: utf-8 -*-


# ***************************************************
# * File        : metrics_classification.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-19
# * Version     : 0.1.031901
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class binary_score(object):

	def __init__(self):
		pass

	# =============================================
	# 准确率
	# =============================================
	def Accuracy(self, y_true, y_pred):
		acc = accuracy_score(y_true, y_pred, normalize = True, sample_weight = None)

		return acc
	# =============================================
	# 混淆矩阵
	# =============================================
	def Confusion_Matrix(self, y_true, y_pred):
		cm = confusion_matrix(y_true, y_pred)
		tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

		return cm, tn, fp, fn, tp
	# ---------------------------------
	# TN, FP, FN, TP
	# ---------------------------------
	def TN(self, y_true, y_pred):
		tn = confusion_matrix(y_true, y_pred)[0, 0]

		return tn

	def FP(self, y_true, y_pred):
		fp = confusion_matrix(y_true, y_pred)[0, 1]

		return fp

	def FN(self, y_true, y_pred):
		fn = confusion_matrix(y_true, y_pred)[1, 0]

		return fn

	def TP(self, y_true, y_pred):
		tp = confusion_matrix(y_true, y_pred)[1, 1]

		return tp

	def TP_TN_FP_FN(self, TP, TN, FP, FN):
		scoring = {
			"TP": make_scorer(TP),
			"TN": make_scorer(TN),
			"FP": make_scorer(FP),
			"FN": make_scorer(FN)
		}

		return scoring
	# ---------------------------------
	# Precision, Recall, F1 score
	# ---------------------------------
	def Precision(self, y_true, y_pred):
		precision = precision_score(y_true, y_pred)

		return precision

	def Recall(self, y_true, y_pred):
		recall = recall_score(y_true, y_pred)

		return recall

	def F1(self, y_true, y_pred):
		f1 = f1_score(y_true, y_pred)

		return f1

	def precisions_recall_threshold(self, y_true, y_score):
		precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

		def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
			plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
			plt.plot(thresholds, recalls[:-1], "g--", label = "Recall")
			plt.xlabel("Thresholds")
			plt.legend(loc="upper left")
			plt.ylim([0, 1])
			plt.show()
	# =============================================
	# roc-acu
	# =============================================
	def AUC_score(self):
		AUC = roc_auc_score(self.y_true, self.y_pred, average = "", sample_weight = None)

		return AUC


class metric_visual:

	def __init__(self):
		pass

	def plot_confusion_matrix(self, cm, classes, normalize = False, title = "Confusion Matrix", cmap = plt.cm.Blues):
		"""
		Print and plot the confusion matrix.
		"""
		# print the confusion matrix
		if normalize:
			cm = cm.astype('flat') / cm.sum(axis = 1)[:, np.newaxis]
			print("Normalized confusion matrix.")
		else:
			print("Confusion matrix, without normalization.")
		print(cm)
		# plot the confusion matrix
		plt.imshow(cm, interpolation = "nearest", cmap = cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation = 45)
		plt.yticks(tick_marks, classes)
		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.0
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j,
					 i,
					 format(cm[i, j], fmt),
					 horizontalignment = "center",
					 color = "white" if cm[i, j] > thresh else "black")
		plt.xlabel("Predicted label")
		plt.ylabel("True label")
		plt.tight_layout()


	def plot_precision_recall_vs_threshold(self, precisions, recalls, thresholds):
		"""
		plot the precisions and recalls to thresholds
		"""
		plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
		plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
		plt.xlabel("Thresholds")
		plt.legend(loc="upper left")
		plt.ylim([0, 1])


class multi_scores(object):

	def __init__(self):
		pass


def scoring():
	scoring_classifier = {
		'accuracy': accuracy_score,
	}

	return scoring_classifier




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
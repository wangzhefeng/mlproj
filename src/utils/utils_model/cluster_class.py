# -*- coding: utf-8 -*-


class Example(object):

	def __init__(self, name, features, label = None):
		"""
		:param name:
		:param features: 浮点数数组
		:param lable:
		"""
		self.name = name
		self.features = features
		self.label = label

	def dimensionality(self):
		return len(self.features)

	def getFeatures(self):
		return self.features[:]

	def getLabel(self):
		return self.label

	def getName(self):
		return self.name

	def distance(self, other):
		return minkowskiDist(self.features, other.getFeatures(), 2)

	def __str__(self):
		return self.name + ":" + str(self.features) + ":" + str(self.label)


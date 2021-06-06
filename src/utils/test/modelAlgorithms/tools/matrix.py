# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

from itertools import product, chain

class Matrix(object):

	def __init__(self, data):
		self.data = data
		self.shape = (len(data), len(data[0]))

	def row(self, row_no):
		"""
		Get a row of the matrix
		:param row_no: int -- Row number of the matrix
		:return: Matrix
		"""
		return Matrix([self.data[row_no]])
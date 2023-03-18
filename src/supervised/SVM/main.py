
# 概念: 
# 
# * 线性SVC分类器
#     - SVM分类器视为在类别之间拟合可能的最宽的街道(平行虚线), 因此也叫最大间隔分类(large margin classification);
#     * 软间隔分类
#         - 目标是尽可能在保持街道宽阔和限制间隔违例之间找到良好的平衡, 这就是软间隔分类；
#         - Scikit-learn的SVM类中, 可以通过超参数C来控制这个平衡: C值越小, 则街道越宽, 但是间隔违例也会越多；
#         - 如果SVM模型过拟合, 可以试试通过降低C来进行正则化；
# * 非线性SVM分类器
#     - 处理非线性数据集的方法之一是添加更多的特征, 比如多项式特征, 某些情况下, 这可能导致数据集变得线性可分；


import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures


# 1.数据准备


iris = datasets.load_iris()
X_train = iris["data"][:, (2, 3)]
y_train = (iris["target"] == 2).astype(np.float64)
print(X_train.shape)
print(y_train.shape)


# 2.建模


## LinearSVC


svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C = 1, loss = "hinge")),
))
svm_clf.fit(X_train, y_train)
# svm_clf.predict(y_test)


## SVC
# 
# * 对大型训练数据集收敛比较慢；


svc_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "linear", C = 1)),
))
svc_clf.fit(X_train, y_train)
# svc_clf.predict(X_test)


## SGDClassifier
# 
# * 对大型训练数据集收敛比较慢；
# * 对于内存处理不了的大型数据集(核外数据)或在线分类任务, 非常有效；


m = 1
C = 1
sgd_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("sgd_svc", SGDClassifier(loss = "hinge", alpha = 1/(m*C))),
))
sgd_clf.fit(X_train, y_train)
# sgd_clf.predict(X_test)


## 多项式特征
# 
# * 添加多项式特征, 使得数据线性可分；


polynoimal_svm_clf = Pipeline((
    ("poly_features", PolynomialFeatures(degree = 3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C = 10, loss = "hinge")),
))
polynoimal_svm_clf.fit(X_train, y_train)
# polynoimal_svm_clf.predict(X_test)


## 多项式核
# 
# * 多项式核技巧: 产生的结果就跟添加了许多多项式特征, 甚至是非常高阶的多项式特征一样, 但实际上并不需要真的添加. 因为实际没有添加任何特征, 所以就不存在数量爆炸的组合特征；
#     - sklearn.svm.SVC中的`coef0`控制的是模型接受高阶多项式还是低阶多项式影响的程度；


from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel = "poly", degree = 3, coef0 = 1, C = 5)),
))
poly_kernel_svm_clf.fit(X_train, y_train)
# poly_kernel_svm_clf.predict(X_test)






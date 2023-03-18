# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from collections import Counter
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
mpl.rcParams['contour.negative_linestyle'] = 'solid'


class OutlierPreprocessing(object):
    """
    异常值处理类、方法
    """
    def __init__(self, data):
        self.data = data
        self.outliers_index = []
        self.feature_change = []


    def outlier_detect_box(self, n):
        """
        箱型图异常值检测
        Args:
            n: 
        Example:
            data = pd.read.csv("data.csv")
            outliers_to_drop = outlier_detect_box(2)
            data = data.drop(outliers_to_drop, axis = 0).reset_index(drop = True)
        """
        for col in self.data.columns.tolist():
            Q1 = np.percentile(self.data[col], 25)
            Q3 = np.percentile(self.data[col], 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            outlier_list_col = self.data[(self.data[col] < Q1 - outlier_step) | (self.data[col] > Q3 + outlier_step)].index
            self.outliers_index.extend(outlier_list_col)
        self.outliers_index = Counter(self.outliers_index)
        multiple_outliers = [k for k, v in self.outliers_index.items() if v > n]
        print("multiple_outliers: {}".format(multiple_outliers))


    def outlier_detect(self, model, sigma = 3):
        """
        基于预测模型识别异常值
        Args:
            model ([type]): predict y values using model
            sigma (int, optional): [description]. Defaults to 3.
        Params:
            data
        Returns:
            outliers: 异常值索引列表
        """
        # 数据处理
        X = self.data.iloc[:, 0:-1]
        y = self.data.iloc[:, -1]
        # 构建预测模型
        try:
            y_pred = pd.Series(model.predict(X), index = y.index)
        except:
            model.fit(X, y)
            y_pred = pd.Series(model.predict(X), index = y.index)
        # 构造 Z-statistic
        resid = y - y_pred
        mean_resid = resid.mean()
        std_resid = resid.std()
        Z = (resid - mean_resid) / std_resid
        self.outliers_index = Z[abs(Z) > sigma].index
        # 打印结果
        print("{} R2={}".format(model, model.score(X, y)))
        print("{} MSE={}".format(model, mean_squared_error(y, y_pred)))
        print("-" * 100)
        print("mean of residuals: {}".format(mean_resid))
        print("std of residuals: {}".format(std_resid))
        print("-" * 100)
        print("{} outliers:\n{}".format(len(self.outliers_index), self.outliers_index.tolist()))
        return y, y_pred, Z


    def outlier_visual(self, y, y_pred, Z):
        """
        可视化基于预测模型识别的异常值
        """
        plt.figure(figsize = (15, 5))

        ax_131 = plt.subplot(1, 3, 1)
        plt.plot(y, y_pred, ".")
        plt.plot(y.loc[self.outliers_index], y_pred.loc[self.outliers_index], "ro")
        plt.legend(["Accepted", "Outlier"])
        plt.xlabel("y")
        plt.ylabel("y_pred");
        ax_132 = plt.subplot(1, 3, 2)
        plt.plot(y, y - y_pred, ".")
        plt.plot(y.loc[self.outliers_index], y.loc[self.outliers_index] - y_pred.loc[self.outliers_index], "ro")
        plt.legend(["Accepted", "Outlier"])
        plt.xlabel("y")
        plt.ylabel("y - y_pred");
        ax_133 = plt.subplot(1, 3, 3)
        Z.plot.hist(bins = 50, ax = ax_133)
        Z.loc[self.outliers_index].plot.hist(color = "r", bins = 50, ax = ax_133)
        plt.legend(["Accepted", "Outlier"])
        plt.xlabel("z")
        plt.show()


    def outlier_processing(self, limit_value = 10, method = "box_IQR", percentile_limit_set = 90, changed_feature_box = []):
        """
        异常值处理
        Args:
            limit_value: 最小处理样本个数集合,当独立样本大于 limit_value, 认为是连续特征
            method
            percentile_limit_set
            changed_feature_box
        Params:
            data
        """
        feature_cnt = self.data.shape[1]
        #离群点盖帽
        if method == "box_iqr":
            for i in range(feature_cnt):
                if len(pd.DataFrame(self.data.iloc[:, i]).drop_duplicates()) >= limit_value:
                    q1 = np.percentile(np.array(self.data.iloc[:, i]), 25)
                    q3 = np.percentile(np.array(self.data.iloc[:, i]), 75)
                    top = q3 + 1.5 * (q3 - q1)
                    self.data.iloc[:, i][self.data.iloc[:, i] > top] = top
                    self.feature_change.append(i)
        if method == "self_define":
            if len(changed_feature_box) == 0:
                # 当方法选择为自定义,且没有定义changed_feature_box,则全量数据全部按照percentile_limit_set的分位点大小进行截断
                for i in range(feature_cnt):
                    if len(pd.DataFrame(self.data.iloc[:, i]).drop_duplicates()) >= limit_value:
                        q_limit = np.percentile(np.array(self.data.iloc[:, i]), percentile_limit_set)
                        self.data.iloc[:, i][self.data.iloc[:, i]] = q_limit
                        self.feature_change.append(i)
            else:
                # 如果定义了changed_feature_box, 则将changed_feature_box里面的按照box方法, changed_feature_box的feature index按照percentile_limit_set的分位点大小进行截断
                for i in range(feature_cnt):
                    if len(pd.DataFrame(self.data.iloc[:, 1]).drop_duplicates()) >= limit_value:
                        if i in changed_feature_box:
                            q1 = np.percentile(np.array(self.data.iloc[:, i]), 25)
                            q3 = np.percentile(np.array(self.data.iloc[:, i]), 75)
                            top = q3 + 1.5 * (q3 - q1)
                            self.data.iloc[:, i][self.data.iloc[:, i] > top] = top
                            self.feature_change.append(i)
                        else:
                            q_limit = np.percentile(np.array(self.data.iloc[:, i]), percentile_limit_set)
                            self.data.iloc[:, i][self.data.iloc[:, i]] = q_limit
                            self.feature_change.append(i)


# ======================================================
# 数据
# ======================================================
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

blobs_params ={
    'random_state': 0,
    'n_samples': n_inliers,
    'n_features': 2
}
dataset = [
    make_blobs(centers = [[0, 0], [0, 0]], cluster_std = 0.5, **blobs_params)[0],
    make_blobs(centers = [[2, 2], [-2, -2]], cluster_std = [0.5, 0.5], **blobs_params)[0],
    make_blobs(centers = [[2, 2], [-2, -2]], cluster_std = [1.5, 0.3], **blobs_params)[0],
    4.0 * (make_moons(n_samples = n_samples, noise = 0.05, random_state = 0)[0] - np.array([0.5, 0.25])),
    14.0 * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)
]
# ======================================================
# 训练好的模型
# ======================================================
elliptic_envelope = EllipticEnvelope(contamination = outliers_fraction)
one_class_svm = OneClassSVM(nu = outliers_fraction, kernel = 'rbf', gamma = 0.1)
isolation_forest = IsolationForest(behaviour = 'new', contamination = outliers_fraction, random_state = 42)
local_outlier_factor = LocalOutlierFactor(n_neighbors = 35, contamination = outliers_fraction)

anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination = outliers_fraction)),
    ("One-Class SVM", OneClassSVM(nu = outliers_fraction, kernel = "rbf", gamma = 0.1)),
    ("Isolation Forest", IsolationForest(behaviour = 'new', contamination = outliers_fraction, random_state = 42)),
    ("Local Outlier Factor", LocalOutlierFactor(n_neighbors = 35, contamination = outliers_fraction))
]
# ======================================================
#
# ======================================================
xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))
plt.figure(figsize = (len(anomaly_algorithms) * 2 + 3, 12.5))
plt.subplots_adjust(left = 0.02, right = 0.98, bottom = 0.01, top = 0.96, wspace = 0.05, hspace = 0.01)

plot_num = 1
rng = np.random.RandomState(42)
for i_dataset, X in enumerate(dataset):
    X = np.concatenate([X, rng.uniform(low = -6, high = 6, size = (n_outliers, 2))], axis = 0)
    for name, algorithm in anomaly_algorithms:
        stime = time.time()
        algorithm.fit(X)
        etime = time.time()
        plt.subplot(len(dataset), len(anomaly_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size = 18)

        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)

        if name != "Local Outlier Factor":
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, level = [0], linewidths = 2, colors = 'black')

        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(X[:, 0], X[:, 1], s = 10, color = colors[(y_pred + 1) // 2])

        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01,
                 ('%.2fs' % (etime - stime)).lstrip('0'),
                 transform = plt.gca().transAxes,
                 size=15,
                 horizontalalignment='right')
        plot_num += 1
plt.show()




if __name__ == "__main__":
    from sklearn.linear_model import Ridge

    train_data_file = "/Users/zfwang/machinelearning/mlproj/src/utils/data/zhengqi_train.txt"
    train_data = pd.read_csv(train_data_file, sep = "\t", encoding = "utf-8")

    outlier_preprocessing = OutlierPreprocessing(train_data)
    y, y_pred, Z = outlier_preprocessing.outlier_detect(model = Ridge(), sigma = 3)
    outlier_preprocessing.outlier_visual(y = y, y_pred = y_pred, Z = Z)

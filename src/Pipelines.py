# -*- coding: utf-8 -*-


# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# **********************************************


# python libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# global variable
rng = np.random.RandomState(0)


def pipelines_demo():
    # pipeline project
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )
    
    # data
    X, y = load_iris(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rng)

    # pipeline fit
    pipe.fit(X_train, y_train)

    # pipeline predict
    score = accuracy_score(pipe.predict(X_test), y_test)
    print(score)


# 测试代码 main 函数
def main():
    pipelines_demo()


if __name__ == "__main__":
    main()


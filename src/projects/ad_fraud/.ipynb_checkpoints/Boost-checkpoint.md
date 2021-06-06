[TOC]

# LightGBM

## 1.LightGBM 理论

* 分类
* 回归
* 排序

## 2.LightGBM 特点

* LightGBM 支持的数据格式有：CSV, TSV, LibSVM
* LightGBM 支持直接使用类别型特征，无需进行 One-Hot encodig，但是需要将类别型变量转换为 `int` 类型；

## 3.Python API

**3.1 下载安装依赖库**

```shell
$ pip install settools wheel numpy scipy scikit-learn -U
```

**3.2 下载安装 LightGBM 库**

```shell
# for Linux, `glibc >= 2.14` is required
$ pip install lightgbm
```

**3.3 导入 `lightgbm` 库**

```python
import lightgbm as lgb
```

**3.4 模型所需数据形式**

* libsvm file
* tsv file
* csv file
* txt file
* NumPy 2D array(s)
* pandas DataFrame
* H2O DataTable’s Frame
* SciPy sparse matrix
* LightGBM binary file

```python
# libsvm text file
train_data = lgb.Dataset("train.svm.bin")

# numpy array
import numpy as np
data = np.random.rand(500, 10)
label = np.random.randint(2, size = 500)
train_data = lgb.Dataset(data = data, label = label)

# scipy.sparse.csr_matrix array
import scipy
csr = scipy.sparse.csr_matrix((data, (row, col)))
train_data = lgb.Dataset(csr)

# save Dataset into a LightGBM binary file
train_data = lgb.Dataset("train.svm.txt")
train_data.save_binary("train.bin")
```

**3.5 创建验证数据集**

```python
validation_data = train_data.create_valid("validation.svm")

# or

validation_data = lgb.Dataset("validation.svm", reference = train_data)
```

**3.6 设置特征名字和类别型特征**

```python
train_data = lgb.Dataset(data, 
                         label = label, 
                         feature_name = ["", "", ...]
                         categorical_feature = [""])
```


**3.6 设置参数**

```python
param = {
    "num_leaves": 31,
    "objective": "binary"
}
param["metric"] = ["auc", "binary_logloss"]
```


**3.7 Training**

```python
# training
num_round = 1000
bst = lgb.train(param, 
                train_data,
                num_round,
                valid_sets = [validation_data])

# early stopping
# 根据验证数据集，可以利用 early stopping 找到最优的 boosting 迭代轮数
num_round = 1000
bst = lgb.train(param, 
                train_data,
                num_round,
                valid_sets = valid_sets,
                early_stopping_rounds = 5,
                first_metric_only = True)

# save model
bst.save_model("model.txt")
bst.save_model("model.txt", num_iteration = bst.best_iteration)
json_model = bst.dump_model()

# load model
bst = lgb.Booster(model_file = "model.txt")
```

**3.8 交叉验证(Cross-Validation)**

```python
lgb.cv(param, 
       train_data,
       num_round,
       nfold = 5)
```



**Prediction:**

* 利用训练好的模型 bst 做预测：

```python
future_data = np.random.rand(4, 10)
y_pred = bst.predict(data)
```

* 如果在训练的过程中设置了提前结束迭代(early stopping)，可以使用最好的迭代模型来做预测：

```python
future_data = np.random.rand(4, 10)
y_pred = bst.predict(data, num_iteration = bst.best_iteration_)
```


# CatBoost

## **1.CatBoost 理论**

> CatBoost is a fast, scalabel, high performance open-scource gradient boosting on decision trees library;

**Features:**

1. Greate quality without parameter tuning
    - Reduce time spent on parameter tuning, because CatBoost provides great results with default parameters;
2. Categorical features support(支持类别性特征，不需要将类别型特征转换为数字类型)
    - Improve your training results with CastBoost that allows you to use non-numeric factors, instead of having to pre-process your data or spend time and effort turning it to numbers.
3. Fast and scalable GPU version
    - Train your model on a fast implementation of gradient-boosting algorithm for GPU.Use a multi-card configuration for large datasets;
4. Imporved accuracy
    - Reduce overfitting when constructing your models with a novel gradient-boosting scheme;
5. Fast prediction
    - Apply your trained model quickly and efficiently even to latency-critical task using CatBoost's models applier;

## **2.下载依赖库**

```shell
pip install numpy six
```

## **3.下载 CatBoost 库**

```shell
# install catboost
$ pip install catboost
```

## **4.快速开始**

* CatBoostClassifier

```python
import numpy as np
from catboost import CatBoostClassifier, Pool

# initialize data
train_data = np.random.randit(0, 100, size = (100, 10))
train_labels = np.random.randint(0, 2, size = (100))
test_data = catboost_pool = Pool(train_data, train_labels)

# build model
model = CatBoostClassifier(iterations = 2,
                           depth = 2,
                           learning_rate = 1,
                           loss_function = "Logloss",
                           verbose = True)

# train model
model.fit(train_data, train_labels)

# prediction using model
y_pred = model.predict(test_data)
y_pred_proba = model.predict_proba(test_data)
print("class = ", y_pred)
print("proba = ", y_pred_proba)
```

* CatBoostRegressor

```python
import numpy as np
from catboost import CatBoostRegressor, Pool

# initialize data
train_data = np.random.randint(0, 100, size = (100, 10))
train_labels = np.random.randint(0, 100, size = (100))
test_data = np.random.randint(0, 100, size = (50, 10))

# initialize Pool
train_pool = Pool(train_data, train_label, cat_features = [0, 2, 5])
test_pool = Pool(test_data, cat_features = [0, 2, 5])

# build model
model = CatBoostRegressor(iterations = 2, 
                          depth = 2,
                          learning_rate = 1, 
                          loss_function = "RMSE")

# train model
model.fit(train_pool)

# prediction
y_pred = model.predict(test_pool)
print(y_pred)
```

* CatBoost

```python
import numpy as np
from catboost import CatBoost, Pool

# read the dataset
train_data = np.random.randint(0, 100, size = (100, 10))
train_labels = np.random.randint(0, 2, size = (100))
test_data = np.random.randint(0, 100, size = (50, 10))

# init pool
train_pool = Pool(train_data, train_labels)
test_pool = Pool(test_data)

# build model
param = {
    "iterations": 5
}
model = CatBoost(param)

# train model
model.fit(train_pool)

# prediction
y_pred_class = model.predict(test_pool, prediction_type = "Class")
y_pred_proba = model.predict(test_pool, prediction_type = "Probability")
y_pred_raw_vals = model.predict(test_pool, prediction_type = "RawFormulaVal")
print("Class", y_pred_class)
print("Proba", y_pred_proba)
print("Raw", y_pred_raw_valss)
```

## 5.Parameter Config


## 6.Objectives and metrics

* Regression
  - MAE
  - MAPE
  - Poisson
  - Quantile
  - RMSE
  - LogLinQuantile
  - Lq
  - Huber
  - Expectile
  - FairLoss
  - NumErrors
  - SMAPE
  - R2
  - MSLE
  - MedianAbsoluteError
* Classification
  - Logloss
  - CrossEntropy
  - Precision
  - Recall
  - F1
  - BalancedAccuracy
  - BalancedErrorRate
  - MCC
  - Accuracy
  - CtrFactor
  - AUC
  - NormalizedGini
  - BriefScore
  - HingeLoss
  - HammingLoss
  - ZeroOneLoss
  - Kapp
  - WKappa
  - LogLikelihoodOfPrediction
* Multiclassification
  - MultiClass
  - MultiClassOneVsAll
  - Precision
  - Recall
  - F1
  - TotalF1
  - MCC
  - Accuracy
  - HingeLoss
  - HammingLoss
  - ZeroOneLoss
  - Kappa
  - WKappa
  - 
* Ranking












**4.数据可视化**

下载 `ipywidgets` 可视化库：

```shell
# install visualization tools
$ pip install ipywidgets
$ jypyter nbextension enable --py widgetsnbextersion
```
CatBoost 数据可视化介绍：

* [Data Visualization](https://catboost.ai/docs/features/visualization.html)
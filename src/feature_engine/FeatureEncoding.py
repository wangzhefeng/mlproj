import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher
from category_encoders import TargetEncoder
from category_encoders import LeaveOneOutEncoder
from category_encoders import WOEEncoder
import category_encoders as ce


"""
# -----------------------
# 类别性特征重编码方法
# -----------------------
    1.One-Hot Encoding
    2.Target Encoding (目标编码)
    3.Leave-one-out Encoding (留一法编码)
    4.Bayesian Target Encoding (贝叶斯目标编码)
    5.Weight of Evidence(WoE) (证据权重)
    6.Nonlinear PCA (非线性 PCA)
    2.Order Encoding

# -----------------------
# category_encoders 库
# -----------------------
import category_encoders as ce

encoder = ce.BackwardDifferenceEncoder(cols=[...])
encoder = ce.BaseNEncoder(cols=[...])
encoder = ce.BinaryEncoder(cols=[...])
encoder = ce.CatBoostEncoder(cols=[...])
encoder = ce.CountEncoder(cols=[...])
encoder = ce.GLMMEncoder(cols=[...])
encoder = ce.HashingEncoder(cols=[...])
encoder = ce.HelmertEncoder(cols=[...])
encoder = ce.JamesSteinEncoder(cols=[...])
encoder = ce.LeaveOneOutEncoder(cols=[...])
encoder = ce.MEstimateEncoder(cols=[...])
encoder = ce.OneHotEncoder(cols=[...])
encoder = ce.OrdinalEncoder(cols=[...])
encoder = ce.SumEncoder(cols=[...])
encoder = ce.PolynomialEncoder(cols=[...])
encoder = ce.TargetEncoder(cols=[...])
encoder = ce.WOEEncoder(cols=[...])

encoder.fit(X, y)
X_cleaned = encoder.transform(X_dirty)
"""


def oneHotEncoding(data, limit_value = 10):
    """
    One-Hot Encoding: pandas get_dummies
    """
    feature_cnt = data.shape[1]
    class_index = []
    class_df = pd.DataFrame()
    normal_index = []
    for i in range(feature_cnt):
        if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) < limit_value:
            class_index.append(i)
            class_df = pd.concat([class_df, pd.get_dummies(data.iloc[:, i], prefix = data.columns[i])], axis = 1)
        else:
            normal_index.append(i)
    data_update = pd.concat([data.iloc[:, normal_index], class_df], axis = 1)
    return data_update



def one_hot_encoder(feature):
    """
    One-Hot Encoding: sklearn.preprocessing.OneHotEncoder
    """
    enc = OneHotEncoder(categories = "auto")
    encoded_feature = enc.fit_transform(feature)
    return encoded_feature


def order_encoder(feature):
    """
    Ordinal Encoding: sklearn.preprocessing.OrdinalEncoder
    """
    enc = OrdinalEncoder()
    encoded_feats = enc.fit_transform(feature)
    return encoded_feats


def label_encoder(data):
    """
    Label Encoder

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    le = LabelEncoder()
    for c in data.columns:
        if data.dtypes[c] == object:
            le.fit(data[c].astype(str))
            data[c] = le.transform(data[c].astype(str))
    return data


if __name__ == "__main__":
    # order
    le =  LabelEncoder()
    classes = [1, 2, 6, 4, 2]
    new_classes = le.fit_transform(classes)
    print(le.classes_)
    print(new_classes)

    le =  LabelEncoder()
    classes = ["paris", "paris", "tokyo", "amsterdam"]
    new_classes = le.fit_transform(classes)
    print(le.classes_)
    print(new_classes)
    
    # enc = OrdinalEncoder()
    # classes = [1, 2, 6, 4, 2]
    # new_classes = enc.fit_transform(classes)
    # print(enc.classes_)
    # print(new_classes)

    # one-hot
    df = pd.DataFrame({
        "City": ["SF", "SF", "SF", "NYC", "NYC", "NYC", "Seattle", "Seattle", "Seattle"],
        "Rent": [3999, 4000, 4001, 3499, 3500, 3501, 2499, 2500, 2501]
    })
    df["Rent"].mean()
    one_hot_df = pd.get_dummies(df, prefix = "city")
    print(one_hot_df)

    # 虚拟编码
    df = pd.DataFrame({
        "City": ["SF", "SF", "SF", "NYC", "NYC", "NYC", "Seattle", "Seattle", "Seattle"],
        "Rent": [3999, 4000, 4001, 3499, 3500, 3501, 2499, 2500, 2501]
    })
    df["Rent"].mean()
    vir_df = pd.get_dummies(df, prefix = "city", drop_first = True)
    print(vir_df)

    # 效果编码
    df = pd.DataFrame({
        "City": ["SF", "SF", "SF", "NYC", "NYC", "NYC", "Seattle", "Seattle", "Seattle"],
        "Rent": [3999, 4000, 4001, 3499, 3500, 3501, 2499, 2500, 2501]
    })
    df["Rent"].mean()
    vir_df = pd.get_dummies(df, prefix = "city", drop_first = True)
    effect_df = vir_df[3:5, ["city_SF", "city_Seattle"]] = -1
    print(effect_df)


    # 单词特征的特征散列化
    def hash_features(word_list, m):
        output = [0] * m
        for word in word_list:
            index = hash_fcn(word) % m
            output[index] += 1
        return output

    # 带符号的特征散列化
    def hash_features(word_list, m):
        output = [0] * m
        for word in word_list:
            index = hash_fcn(word) % m
            sign_bit = sign_hash(word) % 2
            if sign_bit == 0:
                output[index] -= 1
            else:
                output[index] += 1
        return output

    h = FeatureHasher(n_features = m, input_type = "string")
    f = h.trasnform(df["feat"])

    
    enc = TargetEncoder(cols=['Name_of_col','Another_name']) 
    training_set = enc.fit_transform(X_train, y_train)

    enc = LeaveOneOutEncoder(cols=['Name_of_col','Another_name'])
    training_set = enc.fit_transform(X_train, y_train)

    enc = WOEEncoder(cols=['Name_of_col','Another_name']) 
    training_set = enc.fit_transform(X_train, y_train)
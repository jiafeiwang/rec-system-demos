# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : data_preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def create_demo_dataset(file_path, test_size=0.3):
    demo_data = pd.read_csv(file_path)

    dense_cols = ['I' + str(i) for i in range(1, 14)]
    sparse_cols = ['C' + str(i) for i in range(1, 27)]

    #缺失值填充
    demo_data[dense_cols] = demo_data[dense_cols].fillna(0)
    demo_data[sparse_cols] = demo_data[sparse_cols].fillna('null')

    # scaler
    demo_data[dense_cols] = MinMaxScaler().fit_transform(demo_data[dense_cols])

    # LabelEncoding
    for col in sparse_cols:
        demo_data[col] = LabelEncoder().fit_transform(demo_data[col])

    dense_feature_info = [{'name': f'I{i}', 'idx': i-1} for i in range(1, 14)]
    sparse_feature_info = [
        {'name': f'C{i - 13}', 'idx': i-1, 'onehot_dims': demo_data[f'C{i - 13}'].nunique(), 'embed_dims': 6} for i in
        range(14, 40)]

    # split
    X = demo_data.drop(['label'], axis=1).values
    y = demo_data['label'].values.reshape([-1,1])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    return (dense_feature_info,sparse_feature_info), (X_train, y_train, X_val, y_val)


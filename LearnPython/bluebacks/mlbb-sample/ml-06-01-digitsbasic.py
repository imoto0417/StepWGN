# -*- coding: utf-8 -*-
from sklearn import datasets
import numpy as np

# 手書き数字のデータをロードし、変数digitsに格納
digits = datasets.load_digits()

# 特徴量のセットを変数Xに、ターゲットを変数yに格納
X = digits.data
y = digits.target

# Xとyの次元を表示。それぞれ(1797, 64)と(1797,)となる。
# サンプル数1797, 特徴量の次元64, という意味
print(X.shape)
print(y.shape)

# サンプル数と特徴量の次元の取り出し方法
(n_samples, n_features) = X.shape
print('サンプル数: {0}'.format(n_samples))
print('特徴量の次元: {0}'.format(n_features))

# クラス数の取り出し方法
n_classes = len(np.unique(y))
print('クラス数: {0}'.format(n_classes))

# 0～9の全ての数字に対するそれぞれのサンプル数を表示
for i in range(n_classes):
    print('{0}のサンプル数:{1}'.format(i, len(X[y==i])))

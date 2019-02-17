# -*- coding: utf-8 -*-
from sklearn import datasets, svm
import numpy as np

# 手書き数字のデータをロードし、変数digitsに格納
digits = datasets.load_digits()

# 特徴量のセットを変数Xに、ターゲットを変数yに格納
X = digits.data
y = digits.target

# データの順番を入れ替えるためのランダムなNumPy配列
indices = np.random.permutation(len(X))

# 学習用のデータ。全体から100データを省いたもの
X_train = X[indices[:-100]]
y_train = y[indices[:-100]]

# テスト用のデータ。全体から100データ取り出したもの
X_test  = X[indices[-100:]]
y_test  = y[indices[-100:]]

# 分類用にサポートベクトルマシンを用意
clf = svm.SVC(C=1.0, kernel='linear', decision_function_shape='ovr')

# 学習用データに最適化
clf.fit(X_train, y_train)

# テスト用データを分類器に与え、予測を得る
result = clf.predict(X_test)

# テスト用データ数をtotalに格納
total = len(X_test)
# テスト用ターゲット（正解）と予測が一致した数をsuccessに格納
success = sum(result==y_test)

# 正解率をパーセント表示
print('テスト用データでの正解率')
print(100.0*success/total)

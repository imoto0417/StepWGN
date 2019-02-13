# -*- coding: utf-8 -*-
from sklearn import datasets, svm

# 手書き数字のデータをロードし、変数digitsに格納
digits = datasets.load_digits()

# 特徴量のセットを変数Xに、ターゲットを変数yに格納
X = digits.data
y = digits.target

# 分類用にサポートベクトルマシンを用意
clf = svm.SVC(C=1.0, kernel='linear', decision_function_shape='ovr')
# データに最適化
clf.fit(X, y)

# データを分類器に与え、予測を得る
result = clf.predict(X)

# データ数をtotalに格納
total = len(X)
# ターゲット（正解）と予測が一致した数をsuccessに格納
success = sum(result==y)

# 正解率をパーセント表示
print('正解率')
print(100.0*success/total)

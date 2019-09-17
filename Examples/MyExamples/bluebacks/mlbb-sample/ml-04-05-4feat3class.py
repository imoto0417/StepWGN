# -*- coding: utf-8 -*-
from sklearn import datasets, svm

# アヤメのデータをロードし、変数irisに格納
iris = datasets.load_iris()

# 特徴量のセットを変数Xに、ターゲットを変数yに格納
X = iris.data
y = iris.target

# 分類用にサポートベクトルマシンを用意
clf = svm.SVC(C=1.0, kernel='linear', decision_function_shape='ovr')
# データに最適化
clf.fit(X, y)

# データを分類器に与え、予測を得る
result = clf.predict(X)

print('ターゲット')
print(y)
print('機械学習による予測')
print(result)

# データ数をtotalに格納
total = len(X)
# ターゲット（正解）と予測が一致した数をsuccessに格納
success = sum(result==y)

# 正解率をパーセント表示
print('正解率')
print(100.0*success/total)

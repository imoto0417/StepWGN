# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

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

# 分類用に多層ニューラルネットワークを用意
# 毎回異なる乱数を利用
clf = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000, tol=0.0001, random_state=None)

# 学習用データでのニューラルネットワークの学習
print('学習用データで学習中…')
clf.fit(X_train, y_train)

# テスト用データを分類器に与え、予測を得る
result = clf.predict(X_test)

# データ数をtotalに格納
total = len(X_test)
# ターゲット（正解）と予測が一致した数をsuccessに格納
success = sum(result==y_test)

# 正解率をパーセント表示
print('テスト用データでの正解率')
print(100.0*success/total)

# 損失関数のグラフの軸ラベルを設定
plt.xlabel('time step')
plt.ylabel('loss')

# グラフ縦軸の範囲を0以上と定める
plt.ylim(0, max(clf.loss_curve_))

# 損失関数の時間変化を描画
plt.plot(clf.loss_curve_)

# 描画したグラフを表示
plt.show()

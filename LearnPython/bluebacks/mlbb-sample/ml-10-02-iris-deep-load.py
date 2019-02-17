# -*- coding: utf-8 -*-
import keras
from sklearn import datasets
import numpy as np
import sys

# 学習済ファイルの確認
if len(sys.argv)==1:
    print('使用法: python ml-10-02-iris-deep-load.py 学習済ファイル名.h5')
    sys.exit()
savefile = sys.argv[1]

# アヤメのデータをロードし、変数irisに格納
iris = datasets.load_iris()

# 特徴量のセットを変数Xに、ターゲットを変数yに格納
X = iris.data
y = iris.target

# クラス数の取り出し
n_classes = len(np.unique(y))

# ターゲットyをkeras用の形式に変換
y_keras = keras.utils.to_categorical(y, n_classes)

# 学習済ファイルを読み込んでmodelを作成
model = keras.models.load_model(savefile)

# 予測結果の取得
result = model.predict_classes(X, verbose=0)

# 結果の表示
print('ターゲット')
print(y)
print('ディープラーニングによる予測')
print(result)

# データ数をtotalに格納
total = len(X)
# ターゲット（正解）と予測が一致した数をsuccessに格納
success = sum(result==y)

# 正解率をパーセント表示
print('正解率')
print(100.0*success/total)


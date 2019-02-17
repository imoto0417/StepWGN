# -*- coding: utf-8 -*-
import keras
from keras import backend as K
from sklearn import datasets
import numpy as np
import sys

# 数字画像のサイズ 縦(row)と横(col)
img_rows, img_cols = 8, 8

# 学習済ファイルの確認
if len(sys.argv)==1:
    print('使用法: python ml-10-04-digits-cnn-load.py 学習済ファイル名.h5')
    sys.exit()
savefile = sys.argv[1]

# 学習済ファイルを読み込んでmodelを作成
model = keras.models.load_model(savefile)

# 手書き数字のデータをロードし、変数digitsに格納
digits = datasets.load_digits()

# 特徴量のセットを変数Xに、ターゲットを変数yに格納
X = digits.data
y = digits.target

# サンプル数とクラス数の取り出し
n_samples = X.shape[0]
n_classes = len(np.unique(y))

# データXをCNN用の形式に変換
if K.image_data_format() == 'channels_first':
    X = X.reshape(X.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# ターゲットyをkeras用の形式に変換
y_keras = keras.utils.to_categorical(y, n_classes)

# 予測結果の取得
result = model.predict_classes(X, verbose=0)

# データ数をtotalに格納
total = len(X)
# ターゲット（正解）と予測が一致した数をsuccessに格納
success = sum(result==y)

# 正解率をパーセント表示
print('正解率')
print(100.0*success/total)

# -*- coding: utf-8 -*-
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# 手書き数字のデータをロードし、変数digitsに格納
digits = datasets.load_digits()

# 特徴量のセットを変数Xに、ターゲットを変数yに格納
X = digits.data
y = digits.target

# 手書き数字の画像表現を変数imagesに格納
images = digits.images

# 表示エリアの背景をシルバーにセット
fig = plt.figure()
fig.patch.set_facecolor('silver')

# 0～9の10枚の画像をそれぞれ3枚ずつ、計30枚描画
for i in range(10):
    for j in range(3):
        # 数字iの画像のうち、j枚目を取り出す
        img = images[y==i][j]
        #ランダムに取り出したい場合は下記を有効に
        #img = images[y==i][np.random.randint(0, len(images[y==i]))]
        # 縦5x横6の画像表示エリアのうち、3*i+j+1番目に描画
        plt.subplot(5, 6, 3*i + j + 1)
        # グラフとしての軸は描画しない
        plt.axis('off')
        # 白黒を反転した状態で描画      
        plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
        # 各画像にタイトルを描画
        plt.title('Data {0}'.format(i))

# 画像間に余裕をもたせて描画
plt.tight_layout()

# 描画した内容を画面表示
plt.show()

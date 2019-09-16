# scikit-learn ライブラリの読み込み
import matplotlib.pyplot as plt
from sklearn import datasets

# cmap
Cmaps = ["Greys", "Purples", "Blues", "Greens",
         "Oranges", "Reds", "YlOrBr", "YlOrRd", "OrRd", "PuRd"]

# 手書き文字セットを読み込む
digits = datasets.load_digits()

fig = plt.figure()

# どのようなデータか、確認してみる
for i in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}:
    ax = fig.add_subplot(3, 4, i + 1)
    ax.matshow(digits.images[i], cmap=Cmaps[i])
plt.show()

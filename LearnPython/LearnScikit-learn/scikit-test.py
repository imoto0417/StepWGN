# scikit-learn ライブラリの読み込み
import matplotlib.pyplot as plt
from sklearn import datasets

# 手書き文字セットを読み込む
digits = datasets.load_digits()

# どのようなデータか、確認してみる
plt.matshow(digits.images[9], cmap="Greys")
plt.show()

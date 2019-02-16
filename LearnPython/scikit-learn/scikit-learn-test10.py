# K最近傍法
# K最近傍法(KNeighborsClassifier)によるクラス分類

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

data = np.loadtxt("slt10.csv",       # 読み込みたいファイルのパス
                  delimiter=",",    # ファイルの区切り文字
                  skiprows=0,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                  usecols=(1,2) # 読み込みたい列番号
                 )
label = np.loadtxt("slt10.csv",       # 読み込みたいファイルのパス
                  delimiter=",",    # ファイルの区切り文字
                  skiprows=0,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                  usecols=(0) # 読み込みたい列番号
                 )
print(data)
print(label)

# 学習用のデータと結果を準備する
#learn_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
#learn_label = np.array([0, 1, 1, 0])

# アルゴリズムを指定。K最近傍法を採用
clf = KNeighborsClassifier(n_neighbors=1)

# 学習用のデータと結果を学習する,fit()
#clf.fit(learn_data, learn_label)
clf.fit(data, label)

# テストデータによる予測,predict()
test_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
test_label = clf.predict(test_data)

# テスト結果を評価する,accuracy_score()
print("予測対象：", test_data, ", 予測結果→", test_label)
print("正解率＝", accuracy_score([0, 1, 1, 0], test_label))

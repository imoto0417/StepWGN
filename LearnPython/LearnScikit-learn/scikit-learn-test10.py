# K最近傍法
# K最近傍法(KNeighborsClassifier)によるクラス分類

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# 学習用のデータと結果を読み込む
learn_data = np.loadtxt("slt10_learn.csv",       # 読み込みたいファイルのパス
                  delimiter=",",    # ファイルの区切り文字
                  skiprows=0,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                  usecols=(1,2)     # 読み込みたい列番号
                 )
learn_label = np.loadtxt("slt10_learn.csv",       # 読み込みたいファイルのパス
                  delimiter=",",    # ファイルの区切り文字
                  skiprows=0,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                  usecols=(0)       # 読み込みたい列番号
                 )
# テストデータを読み込む
test_data = np.loadtxt("slt10_test.csv",       # 読み込みたいファイルのパス
                  delimiter=",",    # ファイルの区切り文字
                  skiprows=0,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                  usecols=(0,1)     # 読み込みたい列番号
                 )
print(learn_data)
print(learn_label)

# アルゴリズムを指定。K最近傍法を採用
clf = KNeighborsClassifier(n_neighbors=1)
#clf = SVC(kernel='rbf', C=10, gamma=0.1) # clfはclassificationの略語
# 学習用のデータと結果を学習する,fit()
clf.fit(learn_data, learn_label)

print(learn_data)
print(test_data)

# テストデータによる予測,predict()

#test_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
test_label = clf.predict(test_data)

# テスト結果を評価する,accuracy_score()
print("予測対象：", test_data, ", 予測結果→", test_label)
#print("正解率＝", accuracy_score([0, 1, 1, 0], test_label))

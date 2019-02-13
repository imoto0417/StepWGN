# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neural_network import MLPClassifier

# じゃんけんの手のベクトル形式を格納した配列。入力データとして用いる
# グー [1, 0, 0], チョキ [0, 1, 0], パー [0, 0, 1]
janken_array = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# グー, チョキ, パーの名称を格納した配列
janken_class = ['グー', 'チョキ', 'パー']

# 過去何回分の手を覚えているか
n = 3

# じゃんけんの過去の手の初期化
# 人間の手とコンピュータの手をそれぞれn回分。さらに1回分につき3個の数字が必要
Jprev = np.zeros(3*n*2)

# 過去の手（ベクトル形式）をランダムに初期化
for i in range(2*n):
    j = np.random.randint(0, 3)
    Jprev[3*i:3*i+3] = janken_array[j]

# 現在の手（0～2の整数）をランダムに初期化
j = np.random.randint(0, 3)

# 過去の手（入力データ）をscikit_learn用の配列に変換
Jprev_set = np.array([Jprev])
# 現在の手（ターゲット）をscikit_learn用の配列に変換
jnow_set = np.array([j])

# 三層ニューラルネットワークを定義
clf = MLPClassifier(hidden_layer_sizes=(200, ), random_state=None)

# ランダムな入力でオンライン学習を1回行う。
# 初回の学習では、あり得るターゲット(0, 1, 2)を分類器に知らせる必要がある
clf.partial_fit(Jprev_set, jnow_set, classes=[0, 1, 2])

# プログラム上はグー、チョキ、パーは0, 1, 2に対応するが、
# キー入力は入力のしやすさから1, 2, 3に割り当てる
print('1:グー、2:チョキ、3:パー')

# 対戦結果の初期化
win = 0
draw = 0
lose = 0

try:
    while True:
        try:
            # 入力された数値(1～3)を(0～2)に変換
            j = int(input())-1
        except (SyntaxError, NameError, UnicodeDecodeError, ValueError):
            # エラーが起きたら再度入力させる
            continue

        # 入力が0, 1, 2でなければ再度入力させる
        if j<0 or j>2:
            continue

        # 過去のじゃんけんの手（ベクトル形式）をscikit_learn形式に
        Jprev_set = np.array([Jprev])
        # 現在のじゃんけんの手（0～2の整数）をscikit_learn形式に
        jnow_set = np.array([j])

        # コンピュータが、過去の手から人間の現在の手を予測
        jpredict = clf.predict(Jprev_set)

        # 人間の手
        your_choice = j
        # 予測を元にコンピュータが決めた手
        # 予測がグーならパー、予測がチョキならグー、予測がパーならチョキ
        comp_choice = (jpredict[0] + 2)%3

        # 人間の手とコンピュータの手を画面に表示 
        print('あなた:'+janken_class[your_choice]+
              ', コンピュータ:'+janken_class[comp_choice])

        # 勝敗結果を更新
        if your_choice == comp_choice:
            draw += 1
        elif your_choice == (comp_choice+1)%3:
            lose += 1
        else:
            win += 1

        # 勝敗結果を表示
        print('あなたの勝ち: {0}, 負け: {1}, あいこ: {2}'.format(win, lose, draw))

        # 過去の手（入力データ）と現在の手（ターゲット）とでオンライン学習
        clf.partial_fit(Jprev_set, jnow_set)

        # 過去の手の末尾に現在のコンピュータの手を追加
        Jprev = np.append(Jprev[3:], janken_array[comp_choice])
        # 過去の手の末尾に現在の人間の手を追加
        Jprev = np.append(Jprev[3:], janken_array[your_choice])

except KeyboardInterrupt:
    pass

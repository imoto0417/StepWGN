# -*- coding: utf-8 -*-
from sklearn.linear_model import Perceptron
import numpy as np
from PIL import Image, ImageTk
try:
    import Tkinter as tk
except ImportError: # for Python 3
    import tkinter as tk

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

# 単純パーセプトロンを定義
clf = Perceptron(random_state=None)
# ランダムな入力でオンライン学習を1回行う。
# 初回の学習では、あり得るターゲット(0, 1, 2)を分類器に知らせる必要がある
clf.partial_fit(Jprev_set, jnow_set, classes=[0, 1, 2])

# 勝敗の回数を初期化
win = 0
draw = 0
lose = 0

class Application(tk.Frame):
    # 初期化用関数
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.w = 200
        self.h = 200
        self.create_widgets()

    # GUI部品の初期化
    def create_widgets(self):
        w = self.w
        h = self.h

        # コンピュータの手を表示する領域を初期化
        self.comp_canvas = tk.Canvas(self, width=w, height=h, bg='white')
        self.comp_blank_img = tk.PhotoImage(width=w, height=h)
        self.comp_canvas.create_image((w/2,h/2), image=self.comp_blank_img, state='normal')
        self.comp_canvas.image = self.comp_blank_img
        self.comp_canvas.grid(row=0, column=0, columnspan=3)

        # コンピュータの手の画像の読み込み
        self.comp_gu_img = ImageTk.PhotoImage(image=Image.open('ml-images/comp_gu.png'))
        self.comp_choki_img = ImageTk.PhotoImage(image=Image.open('ml-images/comp_choki.png'))
        self.comp_pa_img = ImageTk.PhotoImage(image=Image.open('ml-images/comp_pa.png'))

        # 人間の手を表示する領域を初期化
        self.human_canvas = tk.Canvas(self, width=w, height=h, bg='white')
        self.human_blank_img = tk.PhotoImage(width=w, height=h)
        self.human_canvas.create_image((w/2,h/2), image=self.human_blank_img, state='normal')
        self.human_canvas.image = self.human_blank_img
        self.human_canvas.grid(row=0, column=3, columnspan=3)

        # 人間の手の画像の読み込み
        self.human_gu_img = ImageTk.PhotoImage(image=Image.open('ml-images/human_gu.png'))
        self.human_choki_img = ImageTk.PhotoImage(image=Image.open('ml-images/human_choki.png'))
        self.human_pa_img = ImageTk.PhotoImage(image=Image.open('ml-images/human_pa.png'))

        # グーボタンの初期化
        self.gu_btn = tk.Button(self, text='グー', command=self.human_gu)
        self.gu_btn.grid(row=1, column=0, columnspan=2)
        # チョキボタンの初期化
        self.choki_btn = tk.Button(self, text='チョキ', command=self.human_choki)
        self.choki_btn.grid(row=1, column=2, columnspan=2)
        # パーボタンの初期化
        self.pa_btn = tk.Button(self, text='パー', command=self.human_pa)
        self.pa_btn.grid(row=1, column=4, columnspan=2)

        # 結果表示領域の初期化
        self.result_canvas = tk.Canvas(self, width=2*w, height=30, bg='white')
        self.result_canvas.grid(row=2, column=0, columnspan=6)

        # クリアボタンの初期化
        self.clear_btn = tk.Button(self, text='集計のクリア', command=self.clear)
        self.clear_btn.grid(row=3, column=4, columnspan=2)

    # グーボタンが押されたときに呼ばれる関数
    def human_gu(self):
        w = self.w
        h = self.h
        # 人間のグー画像を表示
        self.human_canvas.create_image((w/2,h/2), image=self.human_gu_img, state='normal')
        self.comp_canvas.image = self.comp_gu_img
        # グー（整数0）を与えて処理を進める
        self.janken_process(0)

    # チョキボタンが押されたときに呼ばれる関数
    def human_choki(self):
        w = self.w
        h = self.h
        # 人間のチョキ画像を表示
        self.human_canvas.create_image((w/2,h/2), image=self.human_choki_img, state='normal')
        self.comp_canvas.image = self.comp_choki_img
        # チョキ（整数1）を与えて処理を進める
        self.janken_process(1)

    # パーボタンが押されたときに呼ばれる関数
    def human_pa(self):
        w = self.w
        h = self.h
        # 人間のパー画像を表示
        self.human_canvas.create_image((w/2,h/2), image=self.human_pa_img, state='normal')
        self.comp_canvas.image = self.comp_pa_img
        # パー（整数2）を与えて処理を進める
        self.janken_process(2)

    # クリアボタンが押されたときの呼ばれる関数
    def clear(self):
        # 勝敗結果のクリア
        global draw, lose, win
        draw = 0
        lose = 0
        win = 0
        # クリアしたデータで結果の再表示
        result_text = 'あなたの勝ち: {0}, 負け: {1}, あいこ: {2} '.format(win, lose, draw)
        self.result_canvas.delete('all')
        self.result_canvas.create_text(200, 15, text=result_text)

    # じゃんけん用ボタンが押されたときの処理
    def janken_process(self, j):
        w = self.w
        h = self.h
        global Jprev

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

        if comp_choice == 0:
            # コンピュータのグー画像表示
            self.comp_canvas.create_image((w/2,h/2), image=self.comp_gu_img, state='normal')
            self.comp_canvas.image = self.comp_gu_img
        elif comp_choice == 1:
            # コンピュータのチョキ画像表示
            self.comp_canvas.create_image((w/2,h/2), image=self.comp_choki_img, state='normal')
            self.comp_canvas.image = self.comp_choki_img
        else:
            # コンピュータのパー画像表示
            self.comp_canvas.create_image((w/2,h/2), image=self.comp_pa_img, state='normal')
            self.comp_canvas.image = self.comp_pa_img

        # 勝敗結果を更新
        global draw, lose, win
        if your_choice == comp_choice:
            draw += 1
        elif your_choice == (comp_choice+1)%3:
            lose += 1
        else:
            win += 1

        # 勝敗結果を表示
        result_text = 'あなたの勝ち: {0}, 負け: {1}, あいこ: {2}'.format(win, lose, draw)
        self.result_canvas.delete('all')
        self.result_canvas.create_text(200, 15, text=result_text)

        # 過去の手（入力データ）と現在の手（ターゲット）とでオンライン学習
        clf.partial_fit(Jprev_set, jnow_set)

        # 過去の手の末尾に現在のコンピュータの手を追加
        Jprev = np.append(Jprev[3:], janken_array[comp_choice])
        # 過去の手の末尾に現在の人間の手を追加
        Jprev = np.append(Jprev[3:], janken_array[your_choice])

root = tk.Tk()
app = Application(master=root)
app.master.title('じゃんけん(単純パーセプトロン)')
app.mainloop()

import matplotlib.pyplot as plt

year = [1975, 1980, 1985, 1990, 1995, 2000, 2010, 2018]
weight = [30, 15, 25, 55, 62, 58, 35, 90]
hight = [90, 84, 76, 13, 50, 44, 3, 22]

colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]

title_name = 'Graph'
y_name = 'Value'
x_name = 'Year'


# グラフタイトル
plt.title(title_name)
# グラフの軸
plt.xlabel(x_name)
plt.ylabel(y_name)

# 折れ線グラフ
plt.plot(year, weight, label='Weight')
plt.plot(year, hight, label='Hight')
plt.bar(year, weight, label='Weight')
plt.bar(year, hight, label='Hight')

# 円グラフ
# plt.pie(weight, labels=colorlist)
# グラフの凡例

plt.legend()
plt.show()

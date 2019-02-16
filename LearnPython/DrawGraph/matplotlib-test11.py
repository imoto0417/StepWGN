
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
sns.set_style('whitegrid')
sns.set_palette('gray')

year = np.array([1975, 1980, 1985, 1990, 1995, 2000, 2010, 2018])
weight = np.array([30, 15, 25, 55, 62, 58, 35, 90])
height = np.array([90, 84, 76, 13, 50, 44, 3, 22])
diff = weight - height
colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]

x_position = np.arange(len(year))

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.bar(x_position, weight, tick_label=year)

ax1.set_xlabel('Year')
ax1.set_ylabel('Value')
ax1.set_title('Weight')

ax2 = fig.add_subplot(2, 2, 2)
ax2.bar(x_position, height, tick_label=year)

ax2.set_xlabel('Year')
ax2.set_ylabel('Value')
ax2.set_title('Height')

ax3 = fig.add_subplot(2, 1, 2)
ax3.bar(x_position, diff, tick_label=year)

ax3.set_xlabel('Year')
ax3.set_ylabel('Value')
ax3.set_title('Diff')

# show plots
plt.tight_layout()
plt.show()
# 折れ線グラフ

# 円グラフ
# plt.pie(weight, labels=colorlist)
#グラフの凡例

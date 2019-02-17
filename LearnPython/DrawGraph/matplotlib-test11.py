
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

sns.set()
sns.set_style('whitegrid')
sns.set_palette('gray')

year = np.array([75, 80, 85, 90, 95, 00, 10, 18])
weight = np.array([30, 15, 25, 55, 62, 58, 35, 90])
height = np.array([90, 84, 76, 13, 50, 44, 3, 22])
diff = weight - height
colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]

x_position = np.arange(len(year))

fig = plt.figure()
#
ax1 = fig.add_subplot(2, 4, 1)
ax1.bar(x_position, weight, tick_label=year)
ax1.set_xlabel('Year')
ax1.set_ylabel('Value')
ax1.set_title('Weight')
#
ax2 = fig.add_subplot(2, 4, 2)
ax2.bar(x_position, height, tick_label=year)
ax2.set_xlabel('Year')
ax2.set_ylabel('Value')
ax2.set_title('Height')
#
ax3 = fig.add_subplot(1, 2, 2)
ax3.bar(x_position, diff, tick_label=year)
ax3.set_xlabel('Year')
ax3.set_ylabel('Value')
ax3.set_title('Diff')
#
pi = math.pi   #mathモジュールのπを利用
igfont = {'family':'IPAexGothic'}
x = np.linspace(0, 360, 100)  #0から360度までの範囲を100分割したnumpy配列
y = np.sin(x*pi/180)
ax4 = fig.add_subplot(2, 2, 3)
ax4.plot(x, y, label='Sin')
ax4.set_xlabel('Angle',**igfont,fontsize=16)
ax4.set_ylabel('値',**igfont, fontsize=16)
ax4.set_title('Sin',**igfont, fontsize=16)


# show plots
plt.tight_layout(pad=1, w_pad=1, h_pad=1)
plt.show()
# 折れ線グラフ

# 円グラフ
# plt.pie(weight, labels=colorlist)
#グラフの凡例

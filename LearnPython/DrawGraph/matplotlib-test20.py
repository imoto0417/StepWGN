import math
import numpy as np
from matplotlib import pyplot as plt
igfont = {'family':'IPAexGothic'}

pi = math.pi   #mathモジュールのπを利用

x = np.linspace(0, 360, 100)  #0から360度までの範囲を100分割したnumpy配列
y = np.sin(x*pi/180)

#グラフタイトル
plt.title('Sin',**igfont, fontsize=16)
#グラフの軸
plt.xlabel('Angle',**igfont,fontsize=16)
plt.ylabel('値',**igfont, fontsize=16)

plt.plot(x, y, label='Sin')

#グラフの凡例
plt.legend()
plt.show()

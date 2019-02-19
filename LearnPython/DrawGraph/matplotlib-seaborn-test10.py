import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
sns.set_style('whitegrid')
sns.set_palette('gray')

x = np.array(['ERS1', 'ERS2', 'ETR1', 'ETR2', 'EIN4'])
y = np.array([12.0, 3.1, 11.8, 2.9, 6.2])

x_position = np.arange(len(x))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(x_position, y, tick_label=x)
ax.set_xlabel('gene')
ax.set_ylabel('gene expression [log(TPM)]')



# show plots
plt.tight_layout()
plt.show()

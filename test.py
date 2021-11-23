from scipy.stats import zipf
import matplotlib.pyplot as plt
import numpy as np

max_k = 100
min_k = 10
num = 20

k = np.linspace(min_k, max_k, num)

print(k)

for a in [1.3, 2.6]:
    p = zipf.pmf(k, a=a)
    print("{} - {}".format(a, p))
    plt.plot(k, p, label='a={}'.format(a), linewidth=2)

plt.xlabel('k')
plt.ylabel('probability')

plt.legend()
plt.show()
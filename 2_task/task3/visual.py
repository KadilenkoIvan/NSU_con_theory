import numpy as np
import matplotlib.pyplot as plt

plt.plot([1,2,4,7,8,16,20,40],[1,1.921,2.907,5.238,5.567,5.663,5.663,5.666], "r--", label="speed boost with for")
plt.plot([1,2,4,7,8,16,20,40],[1,1.031,1.022,1.017,1.21,2.642,4.017,6.5], "g--", label="speed boost with no for")
plt.plot([1,40],[1,40], label="linear")
plt.legend()
plt.show()

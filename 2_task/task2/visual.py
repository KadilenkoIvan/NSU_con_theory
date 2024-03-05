import numpy as np
import matplotlib.pyplot as plt

plt.plot([1,2,4,7,8,16,20,40],[1,1.695,2.777,3.846,4.347,5.882,6.666,7.142], "r--", label="speed boost")
plt.plot([1,40],[1,40], label="linear")
plt.legend()
plt.show()
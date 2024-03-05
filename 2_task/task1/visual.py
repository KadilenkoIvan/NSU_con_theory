import numpy as np
import matplotlib.pyplot as plt

plt.plot([1,2,4,7,8,16,20,40],[1,1.544,3.890,6.814,7.796,15.592,19.297,33.603], "r--", label="speed boost")
plt.plot([1,40],[1,40], label="linear")
plt.legend()
plt.show()
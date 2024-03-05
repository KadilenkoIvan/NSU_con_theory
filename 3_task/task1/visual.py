import numpy as np
import matplotlib.pyplot as plt

plt.plot([1,2,4,7,8,16,20,40],[1, 2.973673214, 6.030508475, 10.49557522, 12.55026455, 24.45360825, 30.41025641, 41.37209302], "r--", label="speed boost")
plt.plot([1,40],[1,40], label="linear")
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

plt.plot([1,2,4,7,8,16,20,40],[1,1.921,2.907,5.238,5.567,5.663,5.663,5.666], "r--", label="speed boost")
plt.plot([1,40],[1,40], label="linear")
plt.legend()
plt.show()

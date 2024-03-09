import numpy as np
import matplotlib.pyplot as plt

plt.plot([1,2,4,7,8,16,20,40],[1,2.08,3.34,4.57,4.94,5.16,5.88,2.21,], "r--", label="speed boost with for")
plt.plot([1,2,4,7,8,16,20,40],[1,1.29,1.32,3.29,4.95,10.34,10.82,11.81,], "g--", label="speed boost with no for")
plt.plot([1,40],[1,40], label="linear")
plt.legend()
plt.show()

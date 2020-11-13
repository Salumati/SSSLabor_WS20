import numpy as np
import matplotlib.pyplot as plt

x = np.array(range(100))

a = -1.688586
b = 5.168557

y = a * x + b

plt.plot(x, y, '-')
plt.ylabel('some numbers')
plt.show()
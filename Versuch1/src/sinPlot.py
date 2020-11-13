import numpy as np
import matplotlib as plt

X = np.linspace(-np.pi, np.pi, 256)
C,S = np.cos(X), np.sin(X)
# plot
fig,ax = plt.subplots()
ax.plot(X,C)
ax.plot(X,S)

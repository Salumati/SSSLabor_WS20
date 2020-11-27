import pyaudio as audio
import numpy as np
import matplotlib.pyplot as plt


# 1. nehme konstanten Ton auf.
# Der Ton sollte eine hoch genuge Amplitude haben
# und sich gleichmäßig wiederholen


# 2. stelle die Periode des Signals graphisch dar

# bestimme anhand des Plots die Grundperiode
# bestimme die Grundfrequenz
# wie gros ist die Signaldauer (in s)?
# wie gros ist die Abstandsfrequenz (in Hz)
# wie gros ist die Signallaenge (in s)?

# 3. berechne mit der Funktioni numpy.fft.fft()
# die Fouriertransformierte des Signals und stelle sie graphisch dar

array = np.sin(range(0, 10))  # just a placeholder for the actual data

plot1 = plt.plot(array)
plot1.show()
fft = np.fft.fft(array)
# plt.show()
np.fft.fftfreq



n = 1
M = 1
deltaT = 1
f = n / (M * deltaT)

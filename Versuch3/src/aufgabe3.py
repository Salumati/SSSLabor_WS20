import numpy as np
import matplotlib.pyplot as plt

# 1. nehme konstanten Ton auf.
# Der Ton sollte eine hoch genuge Amplitude haben
# und sich gleichmaesig wiederholen


# 2. stelle die Periode des Signals graphisch dar
data = np.genfromtxt("../media/data.csv", delimiter=",", skip_header=50000)

plt.title('Floete')
plt.xlabel('Time')
plt.ylabel('Voltage(V)')

plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

# 15500 gesamt
'''
plt.plot(data[25000:25300], "-r")
plt.savefig('../media/FloeteSignalEinzeln.png', dpi=900)
plt.show()
'''
### bestimme anhand des Plots die Grundperiode
# (etwa 185)(von: 10 bis:195)
# 0,00419501133786848072562358276644 s pro schwingung

### bestimme die Grundfrequenz
# (f=1/t) (238,3783784277291Hz)

### wie gros ist die Signaldauer (in s)?
# 220*1024 = 225280 -> 225280 / 44100 -> 5,1083900226757369614512471655329s gesamt dauer (Signaldauer)
# Einzelne Frame Zeit: 2,2675736961451247165532879818594e-e5 s

### wie gros ist die Abtastfrequenz (in Hz)
# 44100Hz

### wie gros ist die Signallaenge?
M = 225280

### Abtastintervall deltaT (in s)
t = 0.00002267573696145127165  # in s
#

# 3. berechne mit der Funktion numpy.fft.fft()
# die Fouriertransformierte des Signals und stelle sie graphisch dar


mulperiods = data[109:563]
period = data[100:300]
N = data.size
Nhalf = int(data.size / 2)
Nval = np.arange(0, 112640)
fourier = np.fft.fft(data)
absValues = np.abs(fourier[:Nhalf])
freq = np.fft.fftfreq(N)

"""
plt.title('Spektrum')
plt.xlabel('Frequenz in Hz')
plt.ylabel('Amplitude')
plt.plot(Nval[:30000], absValues[:30000])
plt.savefig('../media/Fouriertranformierte.png', dpi=900)
"""


plt.title('Mehrere Schwinungen')
plt.xlabel('Zeit')
plt.ylabel('Spannung in mv')
plt.plot(mulperiods)
plt.savefig('../media/multipleperiods.png', dpi=900)
plt.show()

"""
plt.title('Einzelne Schwingung')
plt.xlabel('Zeit')
plt.ylabel('Spannung in mv')
plt.plot(period)
plt.savefig('../media/singleperiod.png', dpi=900)
plt.show()
"""
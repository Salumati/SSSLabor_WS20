import pyaudio as audio
import numpy as np
import matplotlib.pyplot as plt

# 1. nehme konstanten Ton auf.
# Der Ton sollte eine hoch genuge Amplitude haben
# und sich gleichmäßig wiederholen


# 2. stelle die Periode des Signals graphisch dar
data = np.genfromtxt("../media/data.csv", delimiter=",", skip_header=50000)

plt.title('Floete')
plt.xlabel('Time')
plt.ylabel('Voltage(V)')
plt.yticks(range(-14000, 16000, 2000))
plt.xticks(range(0, 500, 25))

plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

# 15500 gesamt
plt.plot(data[25000:25300], "-r")
plt.savefig('../media/FloeteSignalEinzeln.png', dpi=900)
plt.show()


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
t = 0.00002267573696145127165 # in s
#


# 3. berechne mit der Funktion numpy.fft.fft()
# die Fouriertransformierte des Signals und stelle sie graphisch dar

# plot1 = plt.plot(array)
# plot1.show()
fft = np.fft.fft(data[25000:25500])
plt.title('Fouriertranformierte')
plt.xlabel('Frequenz(Hz)')
plt.ylabel('Amplitude')
plt.grid(True)

print(fft)
#plt.plot(fft)
# amplitude = 2 / 500 * np.abs(fft)
#frequency = np.fft.fftfreq(500)*500*1/()

#########################################
spektrumX = np.zeros(fft.size)
spektrumY = np.zeros(fft.size)

n = 0
for value in fft:
    spektrumX[n] = (n / (t * M))
    spektrumY[n] = fft[n]


plt.savefig('../media/Fouriertranformierte.png', dpi=900)
plt.plot(spektrumX,spektrumY)
plt.show()

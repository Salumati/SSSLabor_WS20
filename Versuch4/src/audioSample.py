# example from moodle, creates an audio signal and displays that signal in a plot
# at least I think this is what it does...
import pyaudio
import numpy as np
import matplotlib.pyplot as plt

FORMAT = pyaudio.paInt16
SAMPLEFREQ = 44100
FRAMESIZE = 1024
NOFRAMES = 220
p = pyaudio.PyAudio()
print('running')

stream = p.open(format=FORMAT,channels=1, rate=SAMPLEFREQ, input=True, frames_per_buffer=FRAMESIZE)
data = stream.read(NOFRAMES*FRAMESIZE)
decoded = np.fromstring(data, 'Int16')

stream.stop_stream()
stream.close()
p.terminate()

a = np.asarray(decoded)
np.savetxt("data.csv", a, delimiter=",")

print('done')

plt.plot(data)
plt.show()
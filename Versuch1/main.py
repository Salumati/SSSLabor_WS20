# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
from natsort import natsorted

dirname = os.path.join(os.path.dirname(__file__), 'data/')
allFiles = natsorted(glob.glob(dirname + "m*.csv"))

arrayAverage = np.zeros(20)
arrayStandardVariance = np.zeros(20)
arrayDistance = np.array([100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400, 430, 460, 490, 520, 550, 580, 610, 640, 670])

index = 0
print("Mittelwert / Standardabweichung bestimmen")
for files in allFiles:
    voltage = np.genfromtxt(files, delimiter=",", skip_header=1000, usecols=([4]))
    arrayStandardVariance[index] = np.std(voltage)
    arrayAverage[index] = voltage.mean()
    print("Datei: %d, Mittelwert: %f, Standardabweichung: %f" % (index + 1, arrayAverage[index], arrayStandardVariance[index]))
    index = index + 1

# Create 4 plots
fig, axes = plt.subplots(nrows=2, ncols=2)

#
# PLOT Average
#
axes[0, 0].plot(arrayDistance, arrayAverage, 'o', markersize=2, color='red')
axes[0, 0].plot(arrayDistance, arrayAverage)
axes[0, 0].set_ylabel("Voltage [V]")
axes[0, 0].set_xlabel("Distance [mm]")
axes[0, 0].set_title("Average measured values")

#
# PLOT Standard Variance
#
axes[0, 1].plot(arrayDistance, arrayStandardVariance, 'o', markersize=2, color='red')
axes[0, 1].plot(arrayDistance, arrayStandardVariance)
axes[0, 1].set_xlabel("Distance [mm]")
axes[0, 1].set_ylabel("Voltage [V]")
axes[0, 1].set_title("Standardabweichung")

#
# TASK 2
#
# Calculations from the DIN A4 paper
#
arrayDistancePaper = np.array([100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400, 430, 460, 490, 520, 550, 580, 610, 640, 670, 700])
arrayVoltagePaper = np.array(
    [1.363, 1.212, 1.078, 0.973, 0.897, 0.8215, 0.7653, 0.6992, 0.6567, 0.6374, 0.5986, 0.5604, 0.5415, 0.5227, 0.5228, 0.5037, 0.4848,
     0.4847, 0.4846, 0.4846, 0.4657])
#
# PLOT
#
axes[1, 0].plot(arrayDistancePaper, arrayVoltagePaper, 'o', markersize=2, color='red')
axes[1, 0].plot(arrayDistancePaper, arrayVoltagePaper, markersize=1, linestyle='-')
axes[1, 0].set_xlabel("Distance [mm]")
axes[1, 0].set_ylabel("Voltage [V]")
axes[1, 0].set_title("Nicht logarithmiert")

arrayDistancePaperLog = np.zeros(21)
arrayVoltagePaperLog = np.zeros(21)
for index in range(0, 21):
    arrayDistancePaperLog[index] = np.log(arrayDistancePaper[index])
    arrayVoltagePaperLog[index] = np.log(arrayVoltagePaper[index])
    print("Normal: %f Log: %f           Normal: %f Log: %f" % (
        arrayDistancePaper[index], arrayDistancePaperLog[index], arrayVoltagePaper[index], arrayVoltagePaperLog[index]))

axes[1, 1].plot(arrayDistancePaperLog, arrayVoltagePaperLog, 'o', markersize=2, color='red')
axes[1, 1].plot(arrayDistancePaperLog, arrayVoltagePaperLog, markersize=1, linestyle='-')
axes[1, 1].set_xlabel("Distance [mm]")
axes[1, 1].set_ylabel("Voltage [V]")
axes[1, 1].set_title("Logarithmiert")

# Averages from Log
avg_voltageLog = np.mean(arrayVoltagePaperLog)
avg_distance = np.mean(arrayDistancePaperLog)

#
# Linear Regression
#
a1 = 0
a2 = 0
for i in range(len(arrayVoltagePaperLog)):
    a1 += (arrayVoltagePaperLog[i] - avg_voltageLog) * (arrayDistancePaperLog[i] - avg_distance)
    a2 += pow((arrayVoltagePaperLog[i] - avg_voltageLog), 2)

a = a1 / a2
b = avg_distance - a * avg_voltageLog
print("Lineare Regression:")
print("Parameter A(Steigung): %f" % a)
print("Parameter B: %f" % b)

#
# TASK 3
#
dinaA4LangDatei = glob.glob(dirname + "dina4l.csv")
dinaA4BreitDatei = glob.glob(dirname + "dina4b.csv")

voltageDinA4LangDatei = np.genfromtxt(dinaA4LangDatei[0], delimiter=",", skip_header=1000, usecols=([4]))
dinaA4LangMidDatei = voltageDinA4LangDatei.mean()
standardAbweichungLangDatei = np.std(voltageDinA4LangDatei)

voltageDinA4BreitDatei = np.genfromtxt(dinaA4BreitDatei[0], delimiter=",", skip_header=1000, usecols=([4]))
dinaA4BreitMidDatei = voltageDinA4BreitDatei.mean()
standardAbweichungBreitDatei = np.std(voltageDinA4BreitDatei)

# Werte vom Handschriftlichen Protokoll:
dina4LangVoltage = 0.6758
dina4BreitVoltage = 0.8859

# Vertrauensbereich 68% = x +- Standardabweichung * t
# t für 68% = 1.84
t68 = 1.84
print("Vertrauensbereich Sicherheit: 68%% bei langer Seite zwischen: %f - %f" %
      ((dina4LangVoltage - standardAbweichungLangDatei * t68),
       (dina4LangVoltage + standardAbweichungLangDatei * t68)))

# Vertrauensbereich 95% = x+- 2*(Standardabweichung) * t
# t für 95% = 12.71
t95 = 12.71
print("Vertrauensbereich Sicherheit: 95%% bei langer Seite zwischen: %f - %f" % (
    (dina4LangVoltage - standardAbweichungLangDatei * t95),
    (dina4LangVoltage + standardAbweichungLangDatei * t95)))

# Fehler des Abstands = e^b * a * x * Fehler von x
fehlerAbstandLang68 = math.e ** b * a * dina4LangVoltage * standardAbweichungLangDatei * t68  # für Sicherheit von 68%, korrekturfaktor fehlt!
fehlerAbstandLang95 = math.e ** b * a * dina4LangVoltage * 2 * standardAbweichungLangDatei * t95  # für Sicherheit von 95%

# AbstandLang = e^b * dina4Voltage^a
abstanddinA4Lang = (math.e ** b) * (dina4LangVoltage ** a)

print("Die länge der langen DinA4 Seite beträgt %f ± %f cm (Vertrauensbereich 68,26%%)" % (
    abstanddinA4Lang / 10, fehlerAbstandLang68 / 10))
print("Die länge der langen DinA4 Seite beträgt %f ± %f cm (Vertrauensbereich 95%%)" % (
    abstanddinA4Lang, fehlerAbstandLang95))

# Teil b
# Fehler breite Seite:
fehlerAbstandBreit68 = math.e ** b * a * dina4BreitVoltage * standardAbweichungBreitDatei * t68
fehlerAbstandBreit95 = math.e ** b * a * dina4BreitVoltage * 2 * standardAbweichungBreitDatei * t95

abstanddinA4Breit = (math.e ** b) * (dina4BreitVoltage ** a)
print("Die länge der breiten DinA4 Seite beträgt %f ± %f cm (Vertrauensbereich 68,26%%)" % (
    abstanddinA4Breit / 10, fehlerAbstandBreit68 / 10))

flaeche = abstanddinA4Breit * abstanddinA4Lang

# Fehlerfortpflanzung:
# f(x) = länge * breite
fehlerFlaeche = math.sqrt((abstanddinA4Breit * fehlerAbstandLang68) ** 2 + (abstanddinA4Lang * fehlerAbstandBreit68) ** 2)
print("Die Fläche des DinA4 Blattes beträgt: %fcm ± %fcm fehlerFlaeche" % (flaeche / 100, fehlerFlaeche / 100))

fig.show()

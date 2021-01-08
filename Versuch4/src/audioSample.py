# -*- coding: utf-8 -*-
import pyaudio
from scipy import signal
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import math

FORMAT = pyaudio.paInt16
SAMPLEFREQ = 44100
FRAMESIZE = 1024
NOFRAMES = 220

ABTINTERVALL = 1 / SAMPLEFREQ  # 0.00002267573696145127165  in s
schwellwert = 1500

# paths
pathMedia = "..\media\\"
pathInput = pathMedia + "Sprachinput\\"
pathTest = pathInput + "training\\"

pathHoch = pathInput + "training\Sarah\hoch_t_sarah_"
pathTief = pathInput + "training\Sarah\\tief_t_sarah_"
pathRechts = pathInput + "training\Sarah\\rechts_t_sarah_"
pathLinks = pathInput + "training\Dominic\links_Ref_"

pathRefHoch = pathInput + "mittel_hoch"
pathRefTief = pathInput + "mittel_tief"
pathRefLinks = pathInput + "mittel_links"
pathRefRechts = pathInput + "mittel_rechts"

def record(name):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=SAMPLEFREQ, input=True, frames_per_buffer=FRAMESIZE)

    print('start')
    data = stream.read(NOFRAMES * FRAMESIZE)
    decoded = np.fromstring(data, 'Int16')

    stream.stop_stream()
    stream.close()
    p.terminate()

    a = np.asarray(decoded)
    print('done')

    saveData(name, a)
    plotDataAndSaveFig(name)


def recordSeveral(name, number):
    for i in range(number):
        record(str(name) + '_' + str(i))


def saveData(name, yData):
    print("saving")
    yAxis = yData
    xAxis = np.arange(0, len(yData), 1) * ABTINTERVALL

    np.savetxt(str(name) + ".csv", tuple(zip(xAxis, yAxis)), delimiter=",")
    print("saved")


def plotDataAndSaveFig(name):
    xAxis, yAxis = getDataFromFile(name)
    print(max(yAxis))
    plt.yticks(np.arange(min(yAxis), max(yAxis), max(int(max(yAxis) / 10), 1)))
    plt.xticks(np.arange(min(xAxis), max(xAxis), max(xAxis) / 10))

    plt.ylabel('Voltage in mv')
    plt.xlabel('time in s')
    plt.plot(xAxis, yAxis)
    plt.savefig(name + '.png', dpi=900)
    plt.show()


def getDataFromFile(name):
    csv = name + ".csv"
    xAxis = np.genfromtxt(csv, delimiter=",", usecols=0)
    yAxis = np.genfromtxt(csv, delimiter=",", usecols=1)
    return xAxis, yAxis


# Cut the signal to voice activation. Length 1 second
def trigger(name):
    xAxis, yAxis = getDataFromFile(name)
    y = np.zeros(SAMPLEFREQ)
    for index in range(len(yAxis)):
        if abs(yAxis[index]) >= schwellwert:
            y = yAxis[index:min(index + SAMPLEFREQ + 1, len(yAxis))]
            break
    while len(y) < SAMPLEFREQ:
        np.concatenate(y, np.zeros(SAMPLEFREQ-len(y)))
        #y = np.pad(y, 1)
        # print("AUFGEFUELLT" + str(len(y)))
    print(len(y))
    saveData(name + "_trigger", y)
    plotDataAndSaveFig(name + "_trigger")


def calculateAmplitudenspektrumAndSaveFig(fileName):
    xData, yData = getDataFromFile(fileName)
    Nhalf = int(xData.size / 2)
    Nval = np.arange(0, Nhalf)
    fourier = np.fft.fft(yData)
    absValues = np.abs(fourier[:Nhalf])

    plt.title('Spektrum')
    plt.xlabel('Frequenz in Hz')
    plt.ylabel('Amplitude')
    plt.plot(Nval[:8000], absValues[:8000])
    plt.savefig('../media/' + fileName + '_Amplitudenspektrum.png', dpi=900)



######
###### WINDOWING
######

def mkchunks(arr, window_function, chunk_size):
    for i in range(0, len(arr) - chunk_size + 1, math.floor(chunk_size / 2)):
        yield np.concatenate([[0] * i, list(window_function(arr[i:i + chunk_size])), [0] * (len(arr) - (i + chunk_size))])


def windowedd_fft(data):
    gauss_window = np.array(signal.gaussian(512, 512 / 4))

    windows = np.array(list(mkchunks(data, lambda d: d * gauss_window, 512)))
    fft = np.fft.rfft(windows).mean(0)
    return fft


def main(fileName):
    x, data = getDataFromFile(fileName)

    data_fft = np.abs(windowedd_fft(data))
    freqs = np.fft.rfftfreq(len(data), 1 / 44100)
    limit = np.argmax(freqs > 1000)

    fig, ax = plt.subplots()
    ax.plot(freqs[1:limit], np.abs(data_fft[1:limit]))
    fig.text(0.5, 0.04, 'Frequenz (Hz)', ha='center', va='center')
    fig.text(0.06, 0.5, 'Amplitude', ha='center', va='center', rotation='vertical')
    plt.savefig('../media/' + fileName + '_Windowing.png', dpi=900)
    plt.show()

    saveData('../media/' + fileName + '_Windowing', data_fft)

def calcMean():
    for n in ["hoch", "tief", "rechts", "links"]:
        print("..\media\Sprachinput\Ref\\" + n + "_Ref_" + str(0) + "_trigger_Windowing_Amplitudenspektrum")
        x0,n0 = getDataFromFile("..\media\Sprachinput\Ref\\" + n + "_Ref_" + str(0) + "_trigger_Windowing")
        x1,n1 = getDataFromFile("..\media\Sprachinput\Ref\\" + n + "_Ref_" + str(1) + "_trigger_Windowing")
        x2,n2 = getDataFromFile("..\media\Sprachinput\Ref\\" + n + "_Ref_" + str(2) + "_trigger_Windowing")
        x3,n3 = getDataFromFile("..\media\Sprachinput\Ref\\" + n + "_Ref_" + str(3) + "_trigger_Windowing")
        x4,n4 = getDataFromFile("..\media\Sprachinput\Ref\\" + n + "_Ref_" + str(4) + "_trigger_Windowing")
        mean = np.mean(np.array([n0, n1, n2, n3, n4]), axis=0)
        saveData("../media/mittel_" + n, mean)
        fig, ax = plt.subplots()
        ax.plot(mean)
        fig.text(0.5, 0.04, 'Frequenz (Hz)', ha='center', va='center')
        fig.text(0.06, 0.5, 'Amplitude', ha='center', va='center', rotation='vertical')
        plt.savefig('../media/' + n + '_Windowing_Amplitudenspektrum_Mean.png', dpi=900)
        plt.show()


def pearson():
    # kovarianz berechnen:
    x1,y1 = getDataFromFile("../media/Sprachinput/mittel_hoch")
    x2,y2 = getDataFromFile("../media/Sprachinput/Ref/hoch_Ref_1_trigger_Windowing")
    cor = np.corrcoef(y1,y2)
    print(cor[0][0])
    print(cor[0][1])
    #0.9125217330177889
    #0.9008457689344428

def kovarianz(data1, data2):
    # 1 Kovarianz berechnen:
    meanData1 = np.mean(data1)
    meanData2 = np.mean(data2)
    return np.mean(np.multiply(np.subtract(data1, meanData1), np.subtract(data2, meanData2)))

def bravPears(data1, data2):
    kov = kovarianz(data1, data2)

    # 2 Korrelations Koeffeizienten berechnen:
    stAbw1 = np.std(data1)
    stAbw2 = np.std(data2)
    return np.divide(kov, np.multiply(stAbw1, stAbw2))

def sprachErk(signal):
    # last part
    hoch = getDataFromFile(pathRefHoch)
    tief = getDataFromFile(pathRefTief)
    links = getDataFromFile(pathRefLinks)
    rechts = getDataFromFile(pathRefRechts)

    koHoch = bravPears(signal, hoch)
    koTief = bravPears(signal, tief)
    koLinks = bravPears(signal, links)
    koRechts = bravPears(signal, rechts)

    maxKor = max(koTief, koHoch, koLinks, koRechts)
    """ 
    if maxKor <= 0.2:
        return "Error, word not found"
        """
    if koHoch == maxKor:
        return "HOCH"
    if koTief == maxKor:
        return "TIEF"
    if koLinks == maxKor:
        return "LINKS"
    if koRechts == maxKor:
        return "RECHTS"
    else:
        print(maxKor)
        return "Error, problem in calculation occured"

def test(testSatz):
    genPath = pathTest + testSatz + "\\"

    endD = "Ref_"
    endS = "t_sarah_"
    if (testSatz == "Dominic"):
        end = endD
    else:
        end = endS

    hoch = "hoch"
    tief = "tief"
    links = "links"
    rechts = "rechts"

    correct = 0
    wrong = 0
    print("start testing for data set " + testSatz)
    for d in [hoch, tief, links, rechts]:
        path = genPath + d + "_"
        cCorr = 0
        cWrong = 0
        print("start testing for " + d)
        for i in range(5):
            datax, datay = getDataFromFile(path + end + str(i) + "_trigger_Windowing")
            res = sprachErk(datay)
            if res == d.upper():
                cCorr += 1
            else:
                cWrong += 1
                print(res)
        print("done with " + d)
        print("number corr: " + str(cCorr) + " " + str(cCorr/5 * 100) + "%")
        print("number wrong:" + str(cWrong) + " " + str(cWrong/5 * 100) + "%")
        correct += cCorr
        wrong += cWrong
    print("\n result for complete test:")
    print("number corr: " + str(correct) + " " + str(correct / 20 * 100) + "%")
    print("number wrong:" + str(wrong) + " " + str(wrong / 20 * 100) + "%")
    print()


# korvarianz test
""" 
print(bravPears([1, 2, 3], [2, 3, 4]))
print(bravPears([1, 2, 3], [1, 2, 3]))
print(bravPears([1, 2, 3], [1, 1, 1]))
print(bravPears([1, 2, 3], [1, 2, 4]))
"""

main("../media/randomVoice/randomVoiceInput_trigger")


# Aufgabe 2 a/b)


"""
for i in range(5):
    #print(i)
    #trigger(pathHoch + str(i))
    #trigger(pathTief + str(i))
    #trigger(pathRechts + str(i))
    trigger(pathLinks + str(i))


"""
""" 
for i in range(5):
    print(i)
    # main(pathHoch + str(i) + "_trigger")
    # main(pathTief + str(i) + "_trigger")
    # main(pathRechts + str(i) + "_trigger")
    # main(pathLinks + str(i) + "_trigger")
"""
#calcMean()

#c
# pearson()


#x0,n0 = getDataFromFile("..\media\Sprachinput\Ref\\" + "hoch" + "_Ref_" + str(0))
#print(x0 + n0)
# plt.plot(getSpectrum('../media/randomVoiceInput_trigger.csv')[:10000])
# plt.show()

#calculateAmplitudenspektrumAndSaveFig('../media/randomVoiceInput')
#calculateAmplitudenspektrumAndSaveFig('../media/randomVoiceInput_trigger')

# windowing('../media/randomVoiceInput_trigger')

# trigger("../media/randomVoiceInput")
# recordSeveral("rechts_t_sarah", 5)
# plotDataAndSaveFig('testRecord')

# Aufgabe 2 d)
# """
test("Dominic")
test("Sarah")
# """


"""
d1 = "..\media\Sprachinput\Ref\hoch_Ref_0"
d2 = "..\media\Sprachinput\Ref\hoch_Ref_1"
trigger(d1)
trigger(d2)
"""

'''

def windowing(fileName):
    xData, yData = getDataFromFile(fileName)
    sampleSize = 512
    halfSampleSize = int(sampleSize / 2)
    window = []
    currentPos = 0
    gauss = np.array(signal.windows.gaussian(sampleSize, sampleSize / 4))
    while yData.size > (currentPos + sampleSize):
        multipliedGauss = yData[currentPos: currentPos + sampleSize] * gauss
        fourierTransformed = np.fft.fft(multipliedGauss)

        window.append(fourierTransformed)
        currentPos = currentPos + halfSampleSize

    # Mitteln
    for i in range(len(window)):
        if i == 0 or i == len(window) - 1:
            print("AAA")
        else:
            start = i * 256
            for index in range(256):
                yData[start + index] = (yData[start + index] + window[i][index]) / 2

    Nhalf = int(xData.size / 2)
    Nval = np.arange(0, Nhalf)
    fourier = np.fft.fft(yData)
    absValues = np.abs(fourier[:Nhalf])

    plt.title('Spektrum')
    plt.xlabel('Frequenz in Hz')
    plt.ylabel('Amplitude')
    plt.plot(Nval, absValues)
    plt.savefig('../media/' + fileName + '_Windowing_Amplitudenspektrum.png', dpi=900)









w = window[0]
array = np.empty(0)
prevWin = np.zeros(halfSampleSize)

for win in window:
    array = np.append(array, np.add(win[:halfSampleSize], prevWin))
    prevWin = win[halfSampleSize:]

plt.plot(array)
plt.show()

iff = np.fft.ifft(array)
plt.plot(iff)
plt.show()
'''

"""
    for i in range(5):
        win = window[i]
        plt.title('Spektrum ' + str(i))
        plt.xlabel('Frequenz in Hz')
        plt.ylabel('Amplitude')
        plt.plot(win)
        plt.show()

sampleFreq = 44096

def getSpectrum(file):
    datax,data = getDataFromFile('../media/randomVoiceInput_trigger')[:sampleFreq]
    return windowed_fft(data)


def windows(arr, window_function, window_size):
    for i in range(0, len(arr) - window_size + 1, np.math.floor(window_size / 2)):
        yield np.concatenate(
            [[0] * i, list(window_function(arr[i:i + window_size])), [0] * (len(arr) - (i + window_size))])


def windowed_fft(data):
    gauss_window = np.array(signal.gaussian(512, 512 / 4))

    window = np.array(list(windows(data, lambda d: d * gauss_window, 512)))

    return np.fft.fft(window).mean(0)
"""

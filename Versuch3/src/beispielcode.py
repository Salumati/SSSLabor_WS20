import numpy as np
import matplotlib.pyplot as mpl


def main():
    data = getData()
    xData, yData = prepareData(data)
    plotData(xData, yData)
    transform(data, xData, yData)


def getData():
    data = np.genfromtxt("../media/Experiment3_Data_mund.csv", delimiter=",", skip_header=7)
    return data


def prepareData(data):
    xData = []
    yData = []
    for line in data:
        print(line)
        xData.append(line[0])
        yData.append(line[1])
    return xData, yData


def plotData(xData, yData):
    mpl.title('Mundharmonika')
    mpl.xlabel('Time(ms)')
    mpl.ylabel('Voltage(V)')
    mpl.grid(True)

    mpl.xticks(np.arange(-25, 25, 0.5))
    mpl.xlim(11, 17)
    mpl.plot(xData, yData, "-r", label='lbl')
    mpl.savefig('MundSignal.png', dpi=900)
    mpl.show()


def transform(data, xData, yData):
    four = np.fft.fft(yData)

    mpl.title('Fouriertransform')
    mpl.ylabel('Amplitude')
    mpl.xlabel('Frequency(Hz)')
    mpl.grid(True)

    f = []
    for index in range(0, len(four), 1):
        f.append(index / (len(four) * 0.000_005))
    f = np.array(f)

    mpl.xlim(0, len(four))
    mpl.ylim(0, 25_000)
    mpl.xticks(np.arange(0, 10_000, 700))

    mpl.plot(f[:len(f) // 2], np.abs(four[:len(four) // 2]))
    mpl.savefig('MundTransformed.png', dpi=900)
    mpl.show()


main()

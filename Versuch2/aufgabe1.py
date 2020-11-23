# -*- coding: utf-8 -*-
import numpy as np
import cv2

dunkelMean = "media/dunkelMean.png"
weissMean = "media/weissMean.png"
grauwertkeil = "media/grauwertkeil/grauwertkeil.png"


#
# General Methods
#
def takePictureAndWrite(image_name, multi=False):
    cap = cv2.VideoCapture(0)
    # loop for picture taking
    while (True):
        ret, frame = cap.read()
        pic = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', pic)
        if multi or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # save picture and cam Settings
    cv2.imwrite(image_name, frame)
    print("Took picture " + image_name)
    getCamSettings(cap)
    # When everything done, release the capture and close window
    cap.release()
    cv2.destroyAllWindows()


def getCamSettings(cap):
    print("framewidth:" + str(cap.get(3)))
    print("frameheight:" + str(cap.get(4)))
    print("--------------------------------")
    print("brightness:" + str(cap.get(10)))
    print("contrast:" + str(cap.get(11)))
    print("saturation:" + str(cap.get(12)))
    print("--------------------------------")
    print("gain:" + str(cap.get(14)))
    print("exposure:" + str(cap.get(15)))
    print("--------------------------------")
    print("whitebalance:" + str(cap.get(17)))


def takeMultiplePictures(name, bild_anzahl):
    takePictureAndWrite(name + "_" + str(0) + ".png")
    for index in range(1, bild_anzahl):
        takePictureAndWrite(name + "_" + str(index) + ".png", True)


#
# Messung 1: Grauwertkeil
#
def readGrauwertKeil(img=grauwertkeil):
    grayImage = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # framewidth:640.0
    # frameheight:480.0
    grayValues = []
    grayValues.append(grayImage[0:480, 0:100])  # dunkelster bereich
    grayValues.append(grayImage[0:480, 110:255])
    grayValues.append(grayImage[0:480, 260:405])
    grayValues.append(grayImage[0:480, 410:545])
    grayValues.append(grayImage[0:480, 555:640])  # hellster bereich
    cv2.imwrite('media/grauwertkeil/schwarz.png', grayValues[0])
    cv2.imwrite('media/grauwertkeil/dunkel_grau.png', grayValues[1])
    cv2.imwrite('media/grauwertkeil/grau.png', grayValues[2])
    cv2.imwrite('media/grauwertkeil/grau_hell.png', grayValues[3])
    cv2.imwrite('media/grauwertkeil/weiss.png', grayValues[4])

    for index in range(len(grayValues)):
        print("Mittelwert von Grau%d: %f        Standardabweichung: %f" % (index, np.mean(grayValues[index]), np.std(grayValues[index])))


#
# Messung 2: Dunkelbild
#
def readDunkelbild():
    darkArray = []
    # Bilder einlesen:
    for n in range(10):
        # Dunkelbilder einlesen und zu graubildern machen
        darkArray.append(np.float32(cv2.imread("media/dunkelbilder/dunkelbild_" + str(n) + ".png", cv2.IMREAD_GRAYSCALE)))

    # pixelweise Mittelwert berechnen:
    meanDunkelBild = np.mean(darkArray, axis=0)

    cv2.imwrite(dunkelMean, meanDunkelBild)

    # Kontrast maximieren:
    v = np.float32(cv2.imread(dunkelMean))
    v = v - np.min(v)
    s = v * (255 / np.max(v))
    cv2.imwrite("media/dunkelContrastMax.png", s)


#
# Messung 3: Weissbild
#
def readWeissbild():
    weissArray = []
    # Bilder einlesen:
    for index in range(10):
        image = cv2.imread("media/weissbilder/weissbild_" + str(index) + ".png", cv2.IMREAD_GRAYSCALE)
        weissArray.append(np.float32(image))

    meanWeissBild = np.mean(weissArray, axis=0)
    cv2.imwrite(weissMean, meanWeissBild)

    # Kontrast maximieren:
    v = np.float32(cv2.imread(weissMean))
    v = v - np.min(v)
    s = v * (255 / np.max(v))

    cv2.imwrite("media/weissContrastMax.png", s)

    meanDunkelBild = np.float32(cv2.imread(dunkelMean, cv2.IMREAD_GRAYSCALE))
    imageSubtracted = meanWeissBild - meanDunkelBild
    cv2.imwrite('media/weissBildMinusDunkelbild.png', imageSubtracted)


def kalibrierung(img):
    # Dunkelbild Teil:
    # Ziehe Dunkelbild von zu korrigierenden Bild img ab.
    dunkel = np.float32(cv2.imread(dunkelMean, cv2.IMREAD_GRAYSCALE))
    imgKor = np.subtract(img, dunkel)
    cv2.imwrite("media/dunkelSubtrahiert.png", imgKor)

    # Weisbild Teil
    # Weisbild einlesen:
    weissBild = np.float32(cv2.imread(weissMean, cv2.IMREAD_GRAYSCALE))

    mittelwertWeissbild = np.mean(weissBild)
    norm_image = weissBild / mittelwertWeissbild

    # cv2.imshow('Weiss normiert', norm_image)
    cv2.imwrite("media/weiss_normiert.png", norm_image)

    # Dividiere korrigiertes bild durch  WeißbildNorm
    imgKor = imgKor / norm_image
    cv2.imwrite("media/grauWertKorrektur.png", imgKor)


#########################################################
# Start
#########################################################
print("start programm:")
# Teil 1    Grauwertkeil
# takePictureAndWrite("grauwertkeil.png")
#readGrauwertKeil()

# Teil 2    Dunkelbild
# takeMultiplePictures("dunkelbild", 10)
#readDunkelbild()

# Teil 3
# takeMultiplePictures("weissbild", 10)
readWeissbild()

#kalibrierung(np.float32(cv2.imread('media/grauwertkeil/grauwertkeil.png', cv2.IMREAD_GRAYSCALE)))

print("END")

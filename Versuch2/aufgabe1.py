# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time

dunkelMean = "dunkelMean.png"
weissMean = "weissMean.png"
grauwertkeil = "grauwertkeil.png"


# General Methods
def takePictureAndWrite(image_name):
    # Our operations on the frame come here
    ret, frame = cap.read()
    cv2.imwrite(image_name, frame)
    # When everything done, release the capture
    print("Took picture " + image_name)
    # getCamSettings()
    cap.release()


def getCamSettings():
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
    for index in range(bild_anzahl):
        takePictureAndWrite(name + "_" + index + ".png")


#### Messung 1: Grauwertkeil ####
def readGrauwertKeil():
    grayImage = cv2.imread(cv2.samples.findFile(grauwertkeil), cv2.IMREAD_GRAYSCALE)
    # TODO: Change sizes
    grayValues = []
    grayValues.append(grayImage[0:100, 0:100])
    grayValues.append(grayImage[0:100, 0:100])
    grayValues.append(grayImage[0:100, 0:100])
    grayValues.append(grayImage[0:100, 0:100])
    grayValues.append(grayImage[0:100, 0:100])
    for index in range(len(grayValues)):
        print("Mittelwert von Grau%d: %f        Standardabweichung: %f" % (index, np.mean(grayValues[index]), np.std(grayValues[index])))


#### Messung 2: Dunkelbild ####
def readDunkelbild():
    darkArray = []
    # Bilder einlesen:
    for n in range(10):
        # Dunkelbilder einlesen und zu graubildern machen
        img = cv2.imread("dunkelbilder/dunkelbild_" + str(n) + ".png", cv2.IMREAD_GRAYSCALE)
        # Dunkelbilderwerte in double umwandeln und in darkArray speichern.

    # pixelweise Mittelwert berechnen:
    meanDunkelBild = np.mean(darkArray, axis=0)
    cv2.imwrite(dunkelMean, meanDunkelBild)

    # Kontrast maximieren:
    v = cv2.imread(dunkelMean)
    s = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
    s = cv2.Laplacian(s, cv2.CV_16S, ksize=3)

    s = cv2.convertScaleAbs(s, alpha=255, beta=0)
    cv2.imshow('Dunkel Kontrast maximiert', s)

    ##cv2.imshow('Original Image', meanDunkelBild)
    ##cv2.imshow('New Image', new_image)
    cv2.imwrite("dunkelContrastMax.png", s)


def kalibrierungDunkel(img):
    # Ziehe Dunkelbild von zu korrigierenden Bild img ab.
    dunkel = cv2.imread(dunkelMean, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("dunkelSubtrahiert.png", np.subtract(img, dunkel))


#### Messung 3: Weissbild ####
def readWeissbild():
    weissArray = []
    # Bilder einlesen:
    for index in range(10):
        image = cv2.imread("weissbilder/weissbild_" + str(index) + ".png", cv2.IMREAD_GRAYSCALE)
        weissArray.append(np.float32(image))

    meanWeissBild = np.mean(weissArray, axis=0)
    cv2.imwrite(weissMean, meanWeissBild)
    meanDunkelBild = cv2.imread(dunkelMean)
    imageSubtracted = meanWeissBild - meanDunkelBild
    cv2.imwrite('weissBildMinusDunkelbild.png', imageSubtracted)


def kalibrierung(img):
    # Ziehe Dunkelbild von zu korrigierenden Bild img ab.
    dunkel = cv2.imread(dunkelMean, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("dunkelSubtrahiert.png", np.subtract(img, dunkel))


    # Weisbild einlesen:
    weisBild = np.float32(cv2.imread(weissMean, cv2.IMREAD_GRAYSCALE))

    norm_image = cv2.normalize(weisBild, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    #

    # Ziehe Dunkelbild von zu korrigierenden Bild img ab.
    dunkel = cv2.imread(dunkelMean, cv2.IMREAD_GRAYSCALE)
    imgKor = np.subtract(img, dunkel)
    print()


#### Messung 4: Pixelfehler ####


#########################################################
# Start
#########################################################
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Teil 1    Grauwertkeil
# takePictureAndWrite("grauwertkeil.png")
readGrauwertKeil()

# Teil 2    Dunkelbild
# takeMultiplePictures("dunkelbild", 10)
readDunkelbild()

# Teil 3
# takeMultiplePictures("weissbild", 10)
# readWeissbild()

# Teil 4
k = cv2.waitKey(0)
print("END")

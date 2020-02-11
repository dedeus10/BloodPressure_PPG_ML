#!/usr/bin/env python3

## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## DEVELOPER: Cesar Abascal
## PROFESSORS: Cesar Augusto Prior and Cesar Rodrigues (Yeah. Its almost an overflow!)
## PROJECT: Implementation of biomedical signal processing methods
## ARCHIVE: Mathematical functions
## DATE: 06/05/2019 - updated @ 26/05/2019
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np


# MATH FUNCTIONS ----------------------------------------------------------

# Squaring values
def squaringValues(signal):
    return np.power(signal,2)
#end-if

# Exponential moving average by abascal. Do not provide delay, but do not calculate first and last values.
def expMovingAverage_abascal(signal, window):
    ssize = len(signal)

    maSum = 0
    expMA = np.zeros(ssize)
    k = int((window-1)/2)

    for i in range(k, ssize-k):
        for j in range(i-k, i+k):
            maSum = maSum + signal[j]
        #end-for
        expMA[i] = maSum/window
        maSum = 0
    #end-for
    
    return expMA
#end-def

# Statistical average
def average(signal):
    ssize = len(signal)

    aSum = 0
    for i in np.arange(ssize):
        aSum = aSum + signal[i]
    
    return aSum/ssize
#end-def

#  Calculates blocks of interest with rejected noise, based on the Elgendi method.
def elgendiRealBOIandPeaks(xAxis, signal, MA, THR1, THR2):
    # realBlocksOfInterest = blocks of interest with rejected noise
    samples = len(signal)

    peakx, peaky = [], []
    xpeakmax, ypeakmax, ypeakmaxMAX = 0, 0, 0
    blockWidth = 0
    realBlocksOfInterest = np.zeros(samples)

    for i in np.arange(samples):
        if(MA[i] > THR1[i]):
            blockWidth += 1
            realBlocksOfInterest[i] = 1

            if(signal[i] > ypeakmax):
                xpeakmax = xAxis[i]
                ypeakmax = signal[i]
                if(ypeakmax > ypeakmaxMAX):
                    ypeakmaxMAX = ypeakmax
            #end-if
        elif(blockWidth>=THR2):
            blockWidth = 0
            peakx.append(float(xpeakmax))
            peaky.append(float(ypeakmax))
            xpeakmax = ypeakmax = 0
        #end-if
    #end-for

    # Set blocks area wave amplitude with the maximum ypeak founded.
    realBlocksOfInterest = realBlocksOfInterest * ypeakmaxMAX

    return realBlocksOfInterest, peakx, peaky
#end-def
#!/usr/bin/env python3

## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## DEVELOPER: Cesar Abascal
## PROFESSORS: Cesar Augusto Prior and Cesar Rodrigues (Yeah. Its almost an overflow!)
## PROJECT: Implementation of biomedical signal processing methods
## ARCHIVE: Functions for plotting graphs.
## DATE: 06/05/2019 - updated @ 09/05/2019
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import matplotlib.pyplot as plt
import numpy as np

# GRAPHICAL FUNCTIONS -----------------------------------------------------

def plot_signal_movingAvarages(x=0, signal=0, MAone=0, MAtwo=0, show=True):
    plt.figure('Signal, MApeak and MAbeat', figsize=(14,6)) # 20,10

    plt.ylabel("Amplitude")
    plt.xlabel("Tempo [s]")
    plt.plot(x, signal, "brown", label="Signal")
    plt.plot(x, MAone, "purple", label="MAone")
    plt.plot(x, MAtwo, "black", label="MAtwo")
    plt.legend(loc="best")
    plt.grid()
    
    if(show):
        plt.show()
    #end-if
#end-def

def plot_signal_realBlocksOfInterest(x=0, signal=0, blocksOfInterest=0, show=True):
    plt.figure('Signal and Systolic peak', figsize=(14,6)) # 20,10

    plt.ylabel("Amplitude")
    plt.xlabel("Tempo [s]")
    plt.plot(x, signal, "brown", label="Signal")
    plt.plot(x, blocksOfInterest, "purple", label="Blocks of interest")
    plt.legend(loc="best")
    plt.grid()
    
    if(show):
        plt.show()
    #end-if
#end-def

def plot_signal_peaks(x=0, signal=0, peakx=0, peaky=0, HRV = 0, show=True):
    plt.figure('Signal and peaks', figsize=(20,10)) # 20,10
    plt.subplot(2,1,1)
    plt.ylabel("Amplitude")
    plt.xlabel("Tempo [s]")
    plt.plot(x, signal, "brown")
    plt.scatter(peakx, peaky)
    plt.grid()


    #endP =  HRV[len(HRV)-1]
    #ax = np.linspace(0, endP, len(HRV), endpoint=True)
    #plt.subplot(2,1,2)
    #plt.title("HRV")
    #plt.xlabel("Tempo [s]")
    #plt.ylabel("R-R time")
    #plt.plot(ax, HRV, "blue")
    #plt.grid()
    
    if(show):
        plt.show()
    #end-if
#end-def
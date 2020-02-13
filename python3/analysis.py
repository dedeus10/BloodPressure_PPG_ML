#!/usr/bin/env python3

#
#--------------------------------------------------------------------------------
#--                                                                            --
#--                 Universidade Federal de Santa Maria                        --
#--                        Centro de Tecnologia                                --
#--                 Curso de Engenharia de Computação                          --
#--                 Santa Maria - Rio Grande do Sul/BR                         --
#--                                                                            --
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Design      :                                                              --
#-- File		:                       	                              	   --
#-- Authors     : Luis Felipe de Deus                                          --
# --Mentors     : Cesar Augusto Prior                                          -- 
#--                                                                            -- 
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Created     : 08 Jan 2020                                                  --
#-- Update      : 11 Fev 2020                                                  --
#--------------------------------------------------------------------------------
#--                              Overview                                      --
#--                                                                            --
#-- Code executed in python3                                                   --
#--------------------------------------------------------------------------------
#

#Import the libraries we need
from processing.readandfilter import *
from processing.mathprocessing import *
from processing.makegraphics import *
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import heartpy as hp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time


# @brief:   This Function execute aquisition of RR-Time
# @param:   peaks is a value in time of happened the R wave
# @return:  RRTime is a list the values of difference in time between the R wave
def getRRTime(peakx):
    RRTime = []
    last = 0
    for i in peakx:
        RRTime.append(i - last)
        last = i
        
    return RRTime       
#end

# @brief:   This Function make graphics from PPG and ECG signals
# @param:   
# @return: void
def doGraphics(xAxis, ECG, PPG):
    plt.figure('ECG and PPG Signals', figsize=(14,6))

    plt.subplot(2,1,1)
    #plt.title("ECG")
    plt.ylabel("amplitude")
    plt.plot(xAxis, ECG, "red")
    #plt.plot(xAxis, PPG, "blue")
    plt.grid()

    plt.subplot(2,1,2)
    plt.title("PPG")
    plt.plot(xAxis, PPG, "blue")
    plt.grid()

    plt.show()    
#end

# @brief:   This Function extract the features from PPG signal
# @param:   sampleRte is the frequency which one the data was collected
# @param:   PPG is the PPG signal
# @return:  void
def getPPGfeatures(sampleRate, PPG):
    wd, m = hp.process(PPG, sampleRate)
    
    #print(wd["peaklist"])

    #plt.figure(figsize=(12,4))
    #call plotter
    #hp.plotter(wd, m)

    
    #display measures computed
    #for measure in m.keys():
     #   print('%s: %f' %(measure, m[measure]))
    
    return m

    
#end

# @brief:   This Function makes a sliding window on PPG signal to get the features among the signal
# @param:   PPGsignal is the array with the PPG signal
# @return:  void
def slidingWindow(PPGsignal):
    windowList = []
    index = 0
    bpm, sdnn, rmssd, sdsd, ibi, sd1, sd2 = [], [], [], [], [], [], []
    SBP = []
    cont = 0

    for i in PPGsignal:
        if(index < 1000):
            windowList.append(i)
            #print(i, index)
            index+=1
        else:
            w = np.asarray(windowList, dtype = np.float32)
            m = getPPGfeatures(200, w)
            
            #display measures computed
            #print('%s: %f' %(measure, m[measure]))


            if(math.isnan(m['bpm'])):
                bpm.append(np.mean(bpm))
            else:
                bpm.append(m['bpm'])
            
            if(math.isnan(m['sdnn'])):
                sdnn.append(np.mean(sdnn))
            else:
                sdnn.append(m['sdnn'])

            if(math.isnan(m['rmssd'])):
                rmssd.append(np.mean(rmssd))
            else:
                rmssd.append(m['rmssd'])

            if(math.isnan(m['ibi'])):
                ibi.append(np.mean(ibi))
            else:
                ibi.append(m['ibi'])

            if(math.isnan(m['sdsd'])):
                sdsd.append(np.mean(sdsd))
            else:
                sdsd.append(m['sdsd'])

            if(math.isnan(m['sd1'])):
                sd1.append(np.mean(sd1))
            else:
                sd1.append(m['sd1'])
            
            if(math.isnan(m['sd2'])):
                sd2.append(np.mean(sd2))
            else:
                sd2.append(m['sd2'])

            if(cont<59136):
                SBP.append(150)
            elif(cont<211120):
                SBP.append(170)
            else:
                SBP.append(170)

            for i in range(100):
                windowList.pop(0)
                index-=1
            #time.sleep(2)
        cont+=1
    print("LISTAS:")
    for a in range(len(bpm)):
        print('BPM[%d]: ' %(a),bpm[a])
    print("COUNTER: ", cont)
    data = [bpm,sdnn,rmssd,ibi,sdsd,sd1,sd2,SBP]
    return createDataFrame(data,bpm)
#end

def createDataFrame(dataI,bpm):
    ft = ['bpm','ibi','sdnn', 'sdsd', 'rmssd', 'sd1', 'sd2', 'SBP']
    #Create a indicator of sample named 's'. s1,s2,s3...sn
    sample = ['s' + str(i) for i in range(1,(len(bpm)+1))]
    #Create the dataFrame of all data (basically is an array 2d)
    data = pd.DataFrame(dataI, index=[*ft], columns=[*sample], dtype = float)

    print(data.head())
    print(data.shape)
    return data.T, ft
#end

def computeMultipleLinearRegression(data, features):
    #Separate the data in X- Features | Y- Labels
    X=data[[*features]]  # Features
    y=data['SBP']  # labels

    '''print('--- Print data in Features ----')
    print(X)

    print('-----Print data in Labels ------')
    print(y)  '''

    # Split dataset into training set and test set in case use the same data for trainning and test
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3) # 70% training and 30% 
    
    regressor = LinearRegression()
    regressor.fit(XTrain, yTrain)

    yPred = regressor.predict(XTest)
    for i in range(0, yPred.shape[0]):
        print("yPred: ", yPred[i], "| yTest: ", yTest[i]," | Erro: ", abs(yPred[i] - yTest[i]))
        time.sleep(1)


#end

# ----------------------------------------- MAIN ----------------------------------------------------------------------------------

# Reading signals
#ecgPSG, pletPSG, annMarksPSG, sps, patientName = getSignals(psg=True)
plet_redUPPG, plet_irUPPG, ecgPSG, pletPSG, annMarksPSG, sps = getSignals(uppg=True, psg=True)
ECG = ecgPSG # The ecg data
#PPG = plet_irUPPG
PPG = plet_redUPPG

# Calculating x axis
nsamples = len(ECG)
xAxis = np.linspace(0, nsamples/sps, nsamples, endpoint=True)

# Filtering signals
#ECG
lowcut = 8 
highcut = 20
order = 2 
ECGf = butter_bandpass_filter_zi(ECG, lowcut, highcut, sps, order)
#PPG
lowcut = 0.5 
highcut = 8
order = 2 
PPGf = butter_bandpass_filter_zi(PPG, lowcut, highcut, sps, order)


#graphics
#doGraphics(xAxis, ECGf, PPGf)

#run the analysis
#getPPGfeatures(200, PPGf)
data, ft = slidingWindow(PPGf)
ft.remove('SBP')

computeMultipleLinearRegression(data, ft)
#ECGf = PPGf

'''
# Squaring PSG ECG signal
ECGfs = squaringValues(ECGf)
#
# W1 = 97ms (19pts @ 200Hz)
W1 = 19 # Nearest odd integer
MAqrs = expMovingAverage_abascal(ECGfs, W1)

# W2 = 611ms (123pts @ 200Hz)
W2 = 123 # Nearest odd integer
MAbeat = expMovingAverage_abascal(ECGfs, W2)

# Statiscal mean of the signal
ECGfsa = average(ECGfs)

# Alpha will be the multiplication of ECGfsa by beta plus MAbeat
beta = 0.08 # Provide by Elgendi
alpha = (beta * ECGfsa) + MAbeat

# Threshold1 will be the sum of each point in MAbeat by alpha
THR1 = MAbeat + alpha # array

# Threshold2 will be the same as W1
THR2 = W1 # scalar

# Getting blocks of interest with rejected noise and peaks.
# realBlocksOfInterest = blocks of interest with rejected noise
realBlocksOfInterest, peakx, peaky = elgendiRealBOIandPeaks(xAxis, ECGf, MAqrs, THR1, THR2)

# Obtain R-R Time and print de HeartRate
RRTime = getRRTime(peakx[1:len(peakx)])

# Ploting ECG filtered signal, MApeak and MAbeat
#plot_signal_movingAvarages(xAxis, ECGf, MAqrs, MAbeat, True)

# Ploting ECF filtered signal and Blocks Of Interest
#plot_signal_realBlocksOfInterest(xAxis, ECGf, realBlocksOfInterest, True)

# Ploting ECF filtered signal and peaks
plot_signal_peaks(xAxis, ECGf, peakx, peaky, show=True)
'''
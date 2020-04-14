#!/usr/bin/env python3

#
#--------------------------------------------------------------------------------
#--                                                                            --
#--                 Federal University of Santa Maria                          --
#--                        Technology Center                                   --
#--                     Computer Engineering                                   --
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
#-- Created     : 10 Mar 2020                                                  --
#-- Update      : 13 Apr 2020                                                  --
#--------------------------------------------------------------------------------
#--                              Overview                                      --
#-- This is a begginer project for try to estimate, with a good accuracy,
# Systolic and Diastolic Blood Pressure (SBP and DBP). This approach uses only
# PPG (photoplethysmograph) signals. The main idea is:
#
# 1. Collect raw data from databases (I will use Queensland dataset) and a 
# private UFSM dataset later.
#
# 2. Pre processing this sinals, using a Butterworth (Bandpass) filter for remove
# noise and artefacts from breath
#
# 3. Processing this signals for extracting features, in time and frequency
# domain, this approach uses a sliding window on the signal, like wait four 
# heart beats and than extract features from this "window"
#
# 4. With a dataframe (matrix) from features x samples we can use AI to
# estimate SBP and DBP
#
# 5. We can use Machine Learning algorithms, regression more especific
# or Deep Learning with Neural Networks, or another approach.
#
# 6. To evaluate our results we can use k-fold cross-validation or split
# the dataframe into some % for training and another part for test
#
# 7. Improve our model with the knowledge which we get with this analysis
# and try again, again, again and while(results not goodResults): tryAgain() :) --
#
#
# Anoter Ideas: Use the first and second derivative of PPG, called VPG and APG
# Velocity and Acceleration, with PPG, VPG and APG for feature extraction
# also we can extract width from the PPG signal or there is some papers
# using Heart Rate Variability (HRV) extracted from PPG instead ECG (really nice)
# or use frequency domain in features.
#
#-- Code executed in python3                                                   --
#--------------------------------------------------------------------------------
#

#Import the libraries which we need
import sys
import io
import os
import csv
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heartpy as hp
from processing.readandfilter import *
from processing.mathprocessing import *
from processing.makegraphics import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# @brief: This Function make graphics from PPG signals raw and filtered
# @param: sample time is the the quantity of time needed per sample of the data 
# @param: PPGr is the data of PPG raw without filter
# @param: PPGf is the data of PPG filtered
# @return: void
def doGraphics(sample_time, PPGr, PPGf):
    ax = np.linspace(0, len(PPGr)/sample_time, len(PPGr), endpoint=True)
    ax2 = np.linspace(0, len(PPGf)/sample_time, len(PPGf), endpoint=True)

    plt.figure('PPG Signal', figsize=(14,6))

    plt.subplot(2,1,1)
    plt.title("Filtered", fontsize = 'large',fontweight = 'bold')
    plt.plot(ax2, PPGf, "blue")
    plt.ylabel("Amplitude", fontsize = 'large',fontweight = 'bold')
    plt.xlabel("time(s)", fontsize = 'large',fontweight = 'bold')
    plt.grid()

    plt.subplot(2,1,2)
    plt.title("Raw", fontsize = 'large',fontweight = 'bold')
    plt.plot(ax, PPGr, "black")
    plt.ylabel("Amplitude", fontsize = 'large',fontweight = 'bold')
    plt.xlabel("time(s)", fontsize = 'large',fontweight = 'bold')
    plt.grid()
        
    plt.show()    
    
#end

# @brief:   This Function extract the features from PPG signal, I use a lib called heart py but is temporary
# @param:   sampleRate is the frequency which one the data was collected like 100 equal to 100Hz
# @param:   PPG is the PPG signal, usually is a piece of PPG signal, the lenngth depends the length of the window
# @return:  return a dict where the index is the name of feature like {'mean', number}
def getPPGfeatures(sampleRate, PPG):
    wd, m = hp.process(PPG, sampleRate)
    #Without that comment the lib will plot the pieces of PPG and you can see how is doing
    '''
    plt.figure(figsize=(12,4))
    #call plotter
    hp.plotter(wd, m)

    '''
    #display measures computed
    #for measure in m.keys():
    #    print('%s: %f' %(measure, m[measure]))
    
    return m

# @brief:   This class is the sliding window, first was a function, but I refactored it because sounds better
# in the slidingWindow we have the method computeWindow which actually compute the window
# @param:   PPG_s is the whole PPG signal 
# @param:   SBPr is an list of SBP references from the signal e.g. 120, 118 
# @param:   DBPr is an list of DBP references from the signal e.g. 60, 72
# @return:  void
class slidingWindow:
    def __init__(self, PPG_s, SBPr, DBPr):
        self.signal = PPG_s
        self.SBPr = SBPr
        self.DBPr = DBPr
        self.sizeOfWindow = 1500    #Lengh of window
        self.sliceWindow = 150      #When the window goes on, we remove a slice of the beginning
        self.SBP = []
        self.DBP = []
        self.features = []
        self.featuresName = ['bpm', 'sdnn','rmssd', 'ibi', 'sdsd', 'sd1', 'sd2', 'SBP', 'DBP']
        
    def computeWindow(self):
        windowList, BPList, = [], []
        #Walks through the signal, where x is the enumerate (0,1,2..) and y the signal
        for x, y in enumerate(self.signal):
            #Sliding Window - Check if the window already is full
            if(len(windowList) < self.sizeOfWindow):
                windowList.append(y)
                BPList.append([self.SBPr[x], self.DBPr[x]])
            #With the window full get the features of this part
            else:
                wl = np.asarray(windowList, dtype = np.float32)
                m = getPPGfeatures(100, wl)
                #All the features are in 'm'
                #Walks through the features name and add in to a temp list
                tmp = []
                for feature in self.featuresName[0:-2]:
                    tmp.append(m[feature])
                #Computa the references of SBP and DBP, which means, with a window of 1000 samples
                #We have 1000 references, so we made a mean value of this references
                k,v = 0,0
                for x,y in BPList:
                    k += x
                    v += y
                #Add in to the temp list this references of SBP and DBP
                tmp.append((k/len(BPList)))
                tmp.append((v/len(BPList)))
                #Now add to featuresList where which line is a list of features and labels
                self.features.append(tmp)

                #Now remove the slicee of the window and repeat
                del windowList[0:self.sliceWindow]
                del BPList[0:self.sliceWindow]
        #At least make a daframe with samples x (features+labels)
        sample = ['s' + str(i) for i in range(1,(len(self.features)+1))]
        self.dataFrame = pd.DataFrame(self.features, index=[*sample], columns=[*self.featuresName], dtype = float)
        print(self.dataFrame)
        

# @brief:   This class will make the MLR algorithm to estimate the value of SBP and DBP
# in the class we have the method compute which compute the AI algorith and show us the results
# and the method classify BP which put the prediction value into a list of some classes using a pattern
# so with this results splitted we can see for how much the AI is get wrong 
# @param:   data is a dataframe with the features x samples
# @param:   features is a list of names from features like mean, sd, rmssd
# @param:   label is a str with SBP or DBP name 
# @return:  void
class MultipleLinearRegression():
    def __init__(self, data, features, label):
        self.x = data[[*features]]  # Features
        self.y = data[label]        # labels
        self.label = label
        self.regressorL = LinearRegression()
        self.BPclass = np.zeros(4)

    def classify_BP(self, prediction):
        if(prediction < 5): self.BPclass[0]+=1
        elif(prediction < 10): self.BPclass[1]+=1
        elif(prediction < 15): self.BPclass[2]+=1
        else: self.BPclass[3]+=1
    def compute(self, test_size):
        # Split dataset into training set and test set in case use the same data for trainning and test
        xTrain, xTest, yTrain, yTest = train_test_split(self.x, self.y, test_size=test_size)
        self.regressorL.fit(xTrain, yTrain)

        yPred = self.regressorL.predict(xTest)

        print("\n#### RUNNING MULTIPLE LINEAR REGRESSON -> %s ####"%(self.label))
        erro, std = 0, []
        for i in range(0, yPred.shape[0]):
            #print("yPred: %.3f"%(yPred[i]), "| yTest: %.3f" %(yTest[i])," | Erro: %.3f" %(abs(yPred[i] - yTest[i])))
            #time.sleep(1)
            erro += abs(yPred[i] - yTest[i])
            std.append(abs(yPred[i] - yTest[i]))
            self.classify_BP(abs(yPred[i] - yTest[i]))
            
        print(" ------- END OF ANALYSIS ------")
        print("N° de testes: ", yPred.shape[0])
        print("Mean Error: %.3f" %(erro/yPred.shape[0]))
        print("Standar Deviation NP: %.3f"%(np.std(std)))
        print("Total: %d" %(sum(self.BPclass)))
        print("Classe 5mmHg: %d "%(self.BPclass[0]), "-> %.2f"%(self.BPclass[0]/(sum(self.BPclass))*100),"%")
        print("Classe 10mmHg: %d "%(self.BPclass[1]), "-> %.2f"%(self.BPclass[1]/(sum(self.BPclass))*100), "%"," %.2f"%((self.BPclass[0]+self.BPclass[1])/(sum(self.BPclass))*100),"%")
        print("Classe 15mmHg: %d "%(self.BPclass[2]), "-> %.2f"%(self.BPclass[2]/(sum(self.BPclass))*100), "%"," %.2f"%((self.BPclass[0]+self.BPclass[1]+self.BPclass[2])/(sum(self.BPclass))*100),"%")
        print("Classe >15mmHg: %d "%(self.BPclass[3]), "-> %.2f"%(self.BPclass[3]/(sum(self.BPclass))*100),"%")



# @brief:   This class will make the SVR algorithm to estimate the value of SBP and DBP
# in the class we have the method compute which compute the AI algorith and show us the results
# and the method classify BP which put the prediction value into a list of some classes using a pattern
# so with this results splitted we can see for how much the AI is get wrong 
# @param:   data is a dataframe with the features x samples
# @param:   features is a list of names from features like mean, sd, rmssd
# @param:   label is a str with SBP or DBP name 
# @return:  void
class SupportVectorRegression():
    def __init__(self, data, features, label):
        self.x = data[[*features]]  # Features
        self.y = data[label]        # labels
        self.label = label
        self.regressorSVR = SVR(kernel='sigmoid')
        self.BPclass = np.zeros(4)

    def classify_BP(self, prediction):
        if(prediction < 5): self.BPclass[0]+=1
        elif(prediction < 10): self.BPclass[1]+=1
        elif(prediction < 15): self.BPclass[2]+=1
        else: self.BPclass[3]+=1

    def compute(self, test_size):
        # Split dataset into training set and test set in case use the same data for trainning and test
        xTrain, xTest, yTrain, yTest = train_test_split(self.x, self.y, test_size=test_size) #
        self.regressorSVR.fit(xTrain, yTrain)
        yPred = self.regressorSVR.predict(xTest)

        print("\n#### RUNNING SUPPORT VECTOR REGRESSON -> %s ####" %(self.label))
        erro,desvio = 0, []
        for i in range(0, yPred.shape[0]):
            #print("yPred: %.3f"%(yPred[i]), "| yTest: %.3f" %(yTest[i])," | Erro: %.3f" %(abs(yPred[i] - yTest[i])))
            #time.sleep(1)
            erro += abs(yPred[i] - yTest[i])
            desvio.append(abs(yPred[i] - yTest[i]))
            self.classify_BP(abs(yPred[i] - yTest[i]))
            
        print(" ------- END OF ANALYSIS ------")
        print("N° de testes: ", yPred.shape[0])
        print("Mean Error: %.3f" %(erro/yPred.shape[0]))
        print("Standar Deviation NP: %.3f"%(np.std(desvio)))
        print("Total: %d" %(sum(self.BPclass)))
        print("Classe 5mmHg: %d "%(self.BPclass[0]), "-> %.2f"%(self.BPclass[0]/(sum(self.BPclass))*100),"%")
        print("Classe 10mmHg: %d "%(self.BPclass[1]), "-> %.2f"%(self.BPclass[1]/(sum(self.BPclass))*100), "%"," %.2f"%((self.BPclass[0]+self.BPclass[1])/(sum(self.BPclass))*100),"%")
        print("Classe 15mmHg: %d "%(self.BPclass[2]), "-> %.2f"%(self.BPclass[2]/(sum(self.BPclass))*100), "%"," %.2f"%((self.BPclass[0]+self.BPclass[1]+self.BPclass[2])/(sum(self.BPclass))*100),"%")
        print("Classe >15mmHg: %d "%(self.BPclass[3]), "-> %.2f"%(self.BPclass[3]/(sum(self.BPclass))*100),"%")


 
# @brief:   This Function will get the path+name of files csv
# @param:   path is the path where the csv files are
# @return:  return a list of str path+filename
def findFiles(path):
    paths = [os.path.join(path, name) for name in os.listdir(path)]
    files = [arq for arq in paths if os.path.isfile(arq)]
    csvs = [arq for arq in files if arq.lower().endswith(".csv")]
    csvs.sort()
    return csvs


# @brief:   This Function will get the entire data from a list of cases in queensland dataset
# @param:   cases is a list with the name of cases which the function will looking for
# @return:  return three list with the data from PPG and the references from SBP and DBP 
def getDataN(cases):
    pletData, SBPData, DBPData = [], [], []
    
    for case in cases:
        path = "data/Queensland_signals/"+case+"/"
        filesCase = findFiles(path)
        for arq in filesCase:
            with open(arq) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        if( (float(row[' SBP'])> 70 ) and (float(row[' DBP ']) > 40 ) ):
                            pletData.append(float(row['PPG']))
                            SBPData.append(float(row[' SBP']))
                            DBPData.append(float(row[' DBP ']))
                    except:
                        #print("Expept")
                        pass
        print(case, "Was read")
    return pletData, SBPData, DBPData


# ----------------------------------------- MAIN ----------------------------------------------------------------------------------

case = []
case.append("case29")
case.append("case28")
case.append("case27")
#case.append("case26")

PPG_Raw, SBPr, DBPr = getDataN(case)
sps = 100

# Filtering signals
#PPG
lowcut = 0.5 
highcut = 8
order = 2 
PPGf = butter_bandpass_filter_zi(PPG_Raw, lowcut, highcut, sps, order)

#Create the window and compute 
window = slidingWindow(PPGf,SBPr,DBPr)
window.computeWindow()

#Remove the labels from the "featuresName"
ft = window.featuresName
ft.remove('SBP')
ft.remove('DBP')
#Create the objects from AI algorithms
SBP_MLR = MultipleLinearRegression(window.dataFrame, ft, 'SBP')
DBP_MLR = MultipleLinearRegression(window.dataFrame,ft,'DBP')
SBP_SVR = SupportVectorRegression(window.dataFrame,ft,'SBP')
DBP_SVR = SupportVectorRegression(window.dataFrame,ft,'DBP')
#Now compute the algorithms (the argument is the size of training and test split)
SBP_MLR.compute(0.4)
DBP_MLR.compute(0.4)

#SBP_SVR.compute(0.3)
#DBP_SVR.compute(0.3)

##Make graph
doGraphics(sps, PPG_Raw, PPGf)


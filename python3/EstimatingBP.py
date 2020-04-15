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
#-- Update      : 14 Apr 2020                                                  --
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
# Features description - available at Heartpy toolkit: https://pypi.org/project/heartpy/
#Time domain:
#    beats per minute (BPM)
#    interbeat interval (IBI)
#    standard deviation of RR intervals (SDNN)
#    standard deviation of successive differences (SDSD)
#    root mean square of successive differences (RMSSD)
#    proportion of successive differences above 20ms (pNN20)
#    proportion of successive differences above 50ms (pNN50)
#    median absolute deviation of RR intervals (MAD)
#    Poincare analysis (SD1, SD2, S, SD1/SD1)
#
#Frequency domain:
#    low frequency component (0.04-0.15Hz), LF
#    high frequency component (0.16-0.5Hz), HF
#    lf/hf ratio, Lf/HF


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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing



# @brief: This Function make graphics from PPG signals raw and filtered
# @param: sample time is the the quantity of time needed per sample of the data 
# @param: signal1 is the data of PPG raw without filter
# @param: signal2 is the data of PPG filtered
# @return: void
def doGraphics(sample_time, signal1, signal2, label1, label2):
    ax = np.linspace(0, len(signal1)/sample_time, len(signal1), endpoint=True)
    ax2 = np.linspace(0, len(signal2)/sample_time, len(signal2), endpoint=True)

    plt.figure('PPG Signal', figsize=(14,6))

    plt.subplot(2,1,1)
    plt.title(label1, fontsize = 'large',fontweight = 'bold')
    plt.plot(ax, signal1, "black")
    plt.ylabel("Amplitude", fontsize = 'large',fontweight = 'bold')
    plt.xlabel("time(s)", fontsize = 'large',fontweight = 'bold')
    plt.grid()

    plt.subplot(2,1,2)
    plt.title(label2, fontsize = 'large',fontweight = 'bold')
    plt.plot(ax2, signal2, "blue")
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
    wd, m = hp.process(PPG, sample_rate = sampleRate, calc_freq = True) 

    #Without that comment the lib will plot the pieces of PPG and you can see how is doing
    '''plt.figure(figsize=(12,4))
    #call plotter
    hp.plotter(wd, m)'''
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
        self.featuresName = ['bpm','sdnn','rmssd','ibi','sdsd','s','sd1','sd2','sd1/sd2','hr_mad','pnn20','pnn50','breathingrate', 'SBP','DBP']
        
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
        

# @brief:   This class create the PCA analysis which show us the variance of the features in PCA's (Not ready yet!!!)
# @param:   nData is a dataframe with the features x samples
# @param:   nft is the number of features which you want see 
# @return:  void
class PrincipalComponentAnalysis():
    def __init__(self, nData, nft):
        self.data = nData
        self.nFeatures = nft
        self.pca = PCA()
        self.bestFeatures = 0
    def getBestFeatures(self):
        return self.bestFeatures

    def compute(self):
        # First center and scale the data
        scaled_data = preprocessing.scale(self.data)
        #Ajust the model and execute
        self.pca.fit(scaled_data)
        #pca_data = self.pca.transform(scaled_data)

        #Create a bar graph of PC's and yours variance
        '''
        per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
        labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
        plt.bar(x=range(1,len(per_var)+1),height=per_var, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal component')
        plt.title('Scree Plot')
        plt.show()'''

        # Determine which the biggest influence on PC1
        loading_scores = pd.Series(self.pca.components_[0], index=ft)
        ## now sort the loading scores based on their magnitude
        sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
        # get the names of the top X features
        self.bestFeatures = sorted_loading_scores[0:self.nFeatures].index.values
        # print the names and their scores (and +/- sign)
        print(loading_scores[self.bestFeatures])


# @brief:   This class will make the AI algorithm to estimate the value of SBP and DBP
# in the class we have the method compute which compute the AI algorithm and show us the results
# and the method classify BP which put the prediction value into a list of some classes using a pattern
# so with this results splitted we can see for how much the AI is get wrong 
# @param:   data is a dataframe with the features x samples
# @param:   features is a list of names from features like mean, sd, rmssd
# @param:   label is a str with SBP or DBP name 
# @param:   algorithmName is just a string from the name of tha algorithm 
# @param:   regressor is the object of sklearn for regressors algorithms
# @return:  void
class RegressionAI():
    def __init__(self, data, features, label, algorithmName, regressor):
        self.x = data[[*features]]  # Features
        self.y = data[label]        # labels
        self.label = label
        self.AIname = algorithmName
        self.regressorL = regressor
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

        print("\n#### RUNNING %s -> %s ####"%(self.AIname, self.label))
        erro, std = 0, []
        for i in range(0, yPred.shape[0]):
            #print("yPred: %.3f"%(yPred[i]), "| yTest: %.3f" %(yTest[i])," | Erro: %.3f" %(abs(yPred[i] - yTest[i])))
            #time.sleep(1)
            erro += abs(yPred[i] - yTest[i])
            std.append(abs(yPred[i] - yTest[i]))
            self.classify_BP(abs(yPred[i] - yTest[i]))
            
       
        print("Number of test: %d"%(sum(self.BPclass)))
        print("Mean Error: %.3f" %(erro/yPred.shape[0]))
        print("Standar Deviation NP: %.3f"%(np.std(std)))
        print("Classe 5mmHg: %d "%(self.BPclass[0]), "-> %.2f"%(self.BPclass[0]/(sum(self.BPclass))*100),"%")
        print("Classe 10mmHg: %d "%(self.BPclass[1]), "-> %.2f"%(self.BPclass[1]/(sum(self.BPclass))*100), "%"," %.2f"%((self.BPclass[0]+self.BPclass[1])/(sum(self.BPclass))*100),"%")
        print("Classe 15mmHg: %d "%(self.BPclass[2]), "-> %.2f"%(self.BPclass[2]/(sum(self.BPclass))*100), "%"," %.2f"%((self.BPclass[0]+self.BPclass[1]+self.BPclass[2])/(sum(self.BPclass))*100),"%")
        print("Classe >15mmHg: %d "%(self.BPclass[3]), "-> %.2f"%(self.BPclass[3]/(sum(self.BPclass))*100),"%")
        print(" ------- END OF ANALYSIS ------")

 
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

#Get rae signals and SBP DBP references
PPG_Raw, SBPr, DBPr = getDataN(case)
sps = 100

# Filtering signals
#PPG
lowcut = 0.5 
highcut = 8
order = 2 
PPGf = butter_bandpass_filter_zi(PPG_Raw, lowcut, highcut, sps, order)

# Compute the first and second derivaive of PPG, VPG (Velocity plethysmogram) and APG (Acceleration plethysmogram)
VPG = np.diff(PPGf, n=1)
APG = np.diff(PPGf, n=2)
#Show signals
#doGraphics(sps, VPG, APG, 'Velocity', 'Acceleration')

#Create the window and compute 
window = slidingWindow(PPGf,SBPr,DBPr)
window.computeWindow()

#Remove the labels from the "featuresName"
ft = window.featuresName

#Create the PCA object to analyse the features
pca = PrincipalComponentAnalysis(window.dataFrame, 5)
pca.compute()

ft.remove('SBP')
ft.remove('DBP')
#ft = pca.getBestFeatures()


#Create the objects from AI algorithms
#Multiple Linear Regression
SBP_MLR = RegressionAI(window.dataFrame,ft,'SBP','MULTIPLE LINEAR REGRESSION', LinearRegression())
DBP_MLR = RegressionAI(window.dataFrame,ft,'DBP','MULTIPLE LINEAR REGRESSION', LinearRegression())
#Support Vector Regression
SBP_SVR = RegressionAI(window.dataFrame,ft,'SBP','SUPPORT VECTOR REGRESSON', SVR(kernel='sigmoid'))
DBP_SVR = RegressionAI(window.dataFrame,ft,'DBP','SUPPORT VECTOR REGRESSON', SVR(kernel='sigmoid'))
#Decision Tree
SBP_DT = RegressionAI(window.dataFrame,ft,'SBP','DECISION TREE', DecisionTreeRegressor(random_state=0))
DBP_DT = RegressionAI(window.dataFrame,ft,'DBP','DECISION TREE', DecisionTreeRegressor(random_state=0))
#Logistical Regression
SBP_LR = RegressionAI(window.dataFrame,ft,'SBP','LOGISTIC REGRESSION', LogisticRegression())
DBP_LR = RegressionAI(window.dataFrame,ft,'DBP','LOGISTIC REGRESSION', LogisticRegression())

#Now compute the algorithms (the argument is the size of training and test split)
SBP_MLR.compute(0.4)
DBP_MLR.compute(0.4)

SBP_DT.compute(0.4)
DBP_DT.compute(0.4)

#SBP_LR.compute(0.4)
#DBP_LR.compute(0.4)

#SBP_SVR.compute(0.3)
#DBP_SVR.compute(0.3)

##Make graph
#doGraphics(sps, PPG_Raw, PPGf, 'Raw', 'Filtered')

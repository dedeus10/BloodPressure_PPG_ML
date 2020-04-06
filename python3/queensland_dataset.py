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
#-- Created     : 10 Mar 2020                                                  --
#-- Update      : 10 Mar 2020                                                  --
#--------------------------------------------------------------------------------
#--                              Overview                                      --
#--                                                                            --
#-- Code executed in python3                                                   --
#--------------------------------------------------------------------------------
#

#Import the libraries which we need
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import csv
import heartpy as hp
from processing.readandfilter import *
from processing.mathprocessing import *
from processing.makegraphics import *
import math
from scipy.fftpack import fft
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time
import statistics
from sklearn.svm import SVR

# @brief: This Function make graphics from PPG and ECG signals
# @param: sample time is the the quantity of time needed per data
# @param: PPGr is the data of PPG raw without filter
# @param: PPGf is the data of PPG filtered
# @return: void
def doGraphics(sample_time, PPGr, PPGf):
    ax = np.linspace(0, sample_time, len(PPGf), endpoint=True)

    plt.figure('PPG Signal', figsize=(14,6))

    plt.subplot(2,1,1)
    plt.title("Filtered")
    plt.plot(ax, PPGf, "blue")
    plt.grid()

    plt.subplot(2,1,2)
    plt.title("Raw")
    plt.plot(ax, PPGr, "black")
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
    #    print('%s: %f' %(measure, m[measure]))
    
    return m

# @brief:   This Function makes a sliding window on PPG signal to get the features among the signal
# @param:   PPGsignal is the array with the PPG signal
# @param:   SBPr is the reference from Systolic Blood Pressure
# @param:   DBPr is the reference from Diastolic Blood Pressure
# @return:  void
def slidingWindow(PPGsignal, SBPr, DBPr):
    windowList, SBPlist, DBPlist = [], [], []
    index = 0
    bpm, sdnn, rmssd, sdsd, ibi, sd1, sd2 = [], [], [], [], [], [], []
    SBP, DBP = [], []
    cont, lastSBP, lastDBP = 0,0,0
    sizeOfSlideWindow = 700
    for i in PPGsignal:
        if(index < sizeOfSlideWindow):
            windowList.append(i)
            #print(i, index)
            #print("SBP[%d]: "%(cont),SBPr[cont])
            #Try get de annotaion of SBP
            try:
                SBPlist.append(float(SBPr[cont]))
                lastSBP = float(SBPr[cont])
            except:
                #print("Except SBP")
                SBPlist.append(lastSBP)

            #Try get de annotaion of DBP    
            try:
               #print((DBPr[cont]))
                DBPlist.append(float(DBPr[cont]))
                lastDBP = float(DBPr[cont])
            except:
                #print("Except DBP")
                DBPlist.append(lastDBP)
                

            index+=1
        else:
            w = np.asarray(windowList, dtype = np.float32)
            m = getPPGfeatures(100, w)
            
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

            totalValue = 0
            for x in SBPlist:
                totalValue += x 
            SBP.append((totalValue/len(SBPlist)))

            totalValue = 0
            for y in DBPlist:
                totalValue += y 
            DBP.append((totalValue/len(DBPlist)))

        
            for i in range(100):
                windowList.pop(0)
                SBPlist.pop(0)
                DBPlist.pop(0)
                index-=1
            #time.sleep(2)
           
        cont+=1

   # print("LISTAS:")
   # for a in range(len(bpm)):
   #     print('BPM[%d]: ' %(a),bpm[a])
    print("COUNTER: ", cont)
    data = [bpm,sdnn,rmssd,ibi,sdsd,sd1,sd2,DBP]
    return createDataFrame(data,bpm)
#end

# @brief:   This Function creates a dataframe where features are in the rows and samples in de col
# @param:   DataI is a list with all lists of features
# @param:   bpm is one list random to get the size 
# @return:  data.T the transpost of data frame and ft is a list with the labels from data frame
def createDataFrame(dataI,bpm):
    ft = ['bpm','ibi','sdnn', 'sdsd', 'rmssd', 'sd1', 'sd2', 'DBP']
    #Create a indicator of sample named 's'. s1,s2,s3...sn
    sample = ['s' + str(i) for i in range(1,(len(bpm)+1))]
    #Create the dataFrame of all data (basically is an array 2d)
    data = pd.DataFrame(dataI, index=[*ft], columns=[*sample], dtype = float)

    print(data)
    print(data.shape)
    return data.T, ft
#end

# @brief:   This Function will make the MLR algorithm to estimate the value of SBP and DBP
# @param:   data is a dataframe with the features x samples
# @param:   features is a list of labels from data 
# @return:  void
def computeMultipleLinearRegression(data, features):
    #Separate the data in X- Features | Y- Labels
    X=data[[*features]]  # Features
    y=data['DBP']  # labels

    print('--- Print data in Features ----')
    print(X)

    print('-----Print data in Labels ------')
    print(y)  

    # Split dataset into training set and test set in case use the same data for trainning and test
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.4) # 70% training and 30% 
    
    regressor = LinearRegression()
    regressor.fit(XTrain, yTrain)

    yPred = regressor.predict(XTest)
    print("\n#### RUNNING MULTIPLE LINEAR REGRESSON ####")
    counter = 0
    erro = 0
    desvio = []
    for i in range(0, yPred.shape[0]):
        print("yPred: %.3f"%(yPred[i]), "| yTest: %.3f" %(yTest[i])," | Erro: %.3f" %(abs(yPred[i] - yTest[i])))
        erro += abs(yPred[i] - yTest[i])
        desvio.append(abs(yPred[i] - yTest[i]))
        counter+=1
        #time.sleep(1)
    print(" ------- END OF ANALYSIS ------")
    print("Counter: ", counter)
    print("Mean Error: %.3f" %(erro/counter))
    print("Standar Deviation: %.3f"%(statistics.stdev(desvio)))
    print("Standar Deviation NP: %.3f"%(np.std(desvio)))
#end

# @brief:   This Function will make the SVM algorithm to estimate the value of SBP and DBP
# @param:   data is a dataframe with the features x samples
# @param:   features is a list of labels from data 
# @return:  void
def computeSupportVectorRegression(data, features):
    X=data[[*features]]  # Features
    y=data['DBP']  # labels

    print('--- Print data in Features ----')
    print(X)

    print('-----Print data in Labels ------')
    print(y)  

    # Split dataset into training set and test set in case use the same data for trainning and test
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.4) # 70% training and 30% 
    
    regressor = SVR(kernel='sigmoid')
    regressor.fit(XTrain, yTrain)
    yPred = regressor.predict(XTest)

    print("\n#### RUNNING SUPPORT VECTOR REGRESSON ####")
    counter = 0
    erro = 0
    desvio = []
    for i in range(0, yPred.shape[0]):
        print("yPred: %.3f"%(yPred[i]), "| yTest: %.3f" %(yTest[i])," | Erro: %.3f" %(abs(yPred[i] - yTest[i])))
        erro += abs(yPred[i] - yTest[i])
        desvio.append(abs(yPred[i] - yTest[i]))
        counter+=1
        #time.sleep(1)
    print(" ------- END OF ANALYSIS ------")
    print("Counter: ", counter)
    print("Mean Error: %.3f" %(erro/counter))
    print("Standar Deviation: %.3f"%(statistics.stdev(desvio)))
    print("Standar Deviation NP: %.3f"%(np.std(desvio)))


#5 Predicting a new result
#y_pred = sc_y.inverse_transform((regressor.predict(sc_X.transform(np.array([[6.5]])))))
#print(y_pred)


# @brief:   This Function will get the data separeted from one case of queensland dataset
# @param:   case str wich one shows the registered case number to be used
# @return:  return three list with the data from PPG and the references from SBP and DBP 
def getDataSepareted(case):
    pletData, SBPData, DBPData = [], [], []
    path = "../../queensland_dataset/"+case+"/fulldata/"
    filemName = "uq_vsd_"+case+"_fulldata_08.csv"
    # Open csv file and get data
    with open(path+filemName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pletData.append(float(row['Pleth']))
            SBPData.append((row['NBP (Sys)']))
            DBPData.append((row['NBP (Dia)']))
    return pletData, SBPData, DBPData
 #end 

# @brief:   This Function will get the entire data from one case of queensland dataset
# @param:   case str wich one shows the registered case number to be used
# @return:  return three list with the data from PPG and the references from SBP and DBP 
def getData(case):
    pletData, SBPData, DBPData = [], [], []
    path = "../../queensland_dataset/"+case+"/fulldata/"
    sample = 1
    filemName = "uq_vsd_"+case+"_fulldata_"
    # Open csv file and get data
    try:
        while True:
            if(sample<10):
                filemName  = "uq_vsd_"+case+"_fulldata_0"+str(sample)+".csv"
            else:
                filemName  = "uq_vsd_"+case+"_fulldata_"+str(sample)+".csv"
            with open(path+filemName) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    pletData.append(float(row['Pleth']))
                    SBPData.append((row['NBP (Sys)']))
                    DBPData.append((row['NBP (Dia)']))
            sample+=1
    except Exception as err:  # não pega MemoryError, SystemExit, KeyboardInterrupt
        print("Handling run-time error:",err)
        raise # sempre suba novamente a exceção não tratada, 
             # para que seja visível
    finally:
        return pletData, SBPData, DBPData


# ----------------------------------------- MAIN ----------------------------------------------------------------------------------
case = str(input("Case (01:32): "))
case = "case"+case
#sampleData = str(input("Sample (1:13): "))

PPG_Raw, SBPr, DBPr = getDataSepareted(case)
#PPG_Raw, SBPr, DBPr = getData(case)
sps = 100
#print(PPG_Raw)
print(len(SBPr))
print(len(DBPr))
# Filtering signals
#PPG
lowcut = 0.5 
highcut = 8
order = 2 
PPGf = butter_bandpass_filter_zi(PPG_Raw, lowcut, highcut, sps, order)
#PPGf = PPG_Raw

#w = np.asarray(PPG_Raw, dtype = np.float32)
#m = getPPGfeatures(sps, w)

#run the analysis 
#for i in DBPr:
#    print(i)
#    print(float(i))   

data, ft = slidingWindow(PPGf,SBPr,DBPr)
ft.remove('DBP')
computeSupportVectorRegression(data, ft)
computeMultipleLinearRegression(data, ft)

#ECGf = PPGf
#Make graph
#doGraphics(sps, PPG_Raw, PPGf)

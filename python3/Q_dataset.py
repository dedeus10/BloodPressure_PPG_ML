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
    ax = np.linspace(0, len(PPGr)/sample_time, len(PPGr), endpoint=True)
    ax2 = np.linspace(0, len(PPGf)/sample_time, len(PPGf), endpoint=True)

    plt.figure('PPG Signal', figsize=(14,6))

    plt.subplot(2,1,1)
    plt.title("Full")
    plt.plot(ax2, PPGf, "blue")
    plt.ylabel("Amplitude")
    plt.xlabel("time(s)")
    plt.grid()

    plt.subplot(2,1,2)
    plt.title("Raw")
    plt.plot(ax, PPGr, "black")
    plt.ylabel("Amplitude")
    plt.xlabel("time(s)")
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
class slidingWindow:
    def __init__(self, PPG_s, SBPr, DBPr):
        self.signal = PPG_s
        self.SBPr = SBPr
        self.DBPr = DBPr
        self.sizeOfWindow = 1000
        self.sliceWindow = 100
        self.SBP = []
        self.DBP = []
        self.features = []
        self.featuresName = ['bpm', 'sdnn','rmssd', 'ibi', 'sdsd', 'sd1', 'sd2', 'SBP', 'DBP']
        
    def computeWindow(self):
        windowList, BPList, = [], []
        #Percorre o sinal, sendo x o enumerate (0,1,2..) e y o sinal
        for x, y in enumerate(self.signal):
            #Sliding Window - Verifica se a janela ja tem o numero de amostras suficientes
            if(len(windowList) < self.sizeOfWindow):
                windowList.append(y)
                BPList.append([self.SBPr[x], self.DBPr[x]])
            #Com o tamanho de amostras suficentes extrai as featurers desta janela
            else:

                wl = np.asarray(windowList, dtype = np.float32)
                m = getPPGfeatures(100, wl)
                #Todas as features estão em 'm'
                #percorre o array de nomes de features e adiciona em uma lista temporaria
                tmp = []
                for feature in self.featuresName[0:-2]:
                    tmp.append(m[feature])
                #Computa as referencias de SBP e DBP nesta janela
                k,v = 0,0
                for x,y in BPList:
                    k += x
                    v += y
                #Adiciona na lista a media das referencias de SBP e DBP
                tmp.append((k/len(BPList)))
                tmp.append((v/len(BPList)))
                #features e uma lista onde cada linha é uma lista com as features e as labels
                self.features.append(tmp)

                #Remove uma parte de elementos das listas para deslocar a janela 
                del windowList[0:self.sliceWindow]
                del BPList[0:self.sliceWindow]
        #Por fim, cria um dataframe contendo amostras x (features+labels)
        sample = ['s' + str(i) for i in range(1,(len(self.features)+1))]
        self.dataFrame = pd.DataFrame(self.features, index=[*sample], columns=[*self.featuresName], dtype = float)
        print(self.dataFrame)
        

# @brief:   This Function will make the MLR algorithm to estimate the value of SBP and DBP
# @param:   data is a dataframe with the features x samples
# @param:   features is a list of labels from data 
# @return:  void
class MultipleLinearRegression():
    def __init__(self, data, features, label):
        self.x = data[[*features]]  # Features
        self.y = data[label]  # labels
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
            erro += abs(yPred[i] - yTest[i])
            std.append(abs(yPred[i] - yTest[i]))
            self.classify_BP(abs(yPred[i] - yTest[i]))
            #time.sleep(1)
        print(" ------- END OF ANALYSIS ------")
        print("N° de testes: ", yPred.shape[0])
        print("Mean Error: %.3f" %(erro/yPred.shape[0]))
        print("Standar Deviation NP: %.3f"%(np.std(std)))
        print("Total: %d" %(sum(self.BPclass)))
        print("Classe 5mmHg: %d "%(self.BPclass[0]), "-> %.2f"%(self.BPclass[0]/(sum(self.BPclass))*100),"%")
        print("Classe 10mmHg: %d "%(self.BPclass[1]), "-> %.2f"%(self.BPclass[1]/(sum(self.BPclass))*100), " % -",((self.BPclass[0]+self.BPclass[1])/(sum(self.BPclass))*100),"%")
        print("Classe 15mmHg: %d "%(self.BPclass[2]), "-> %.2f"%(self.BPclass[2]/(sum(self.BPclass))*100), " % -",((self.BPclass[0]+self.BPclass[1]+self.BPclass[2])/(sum(self.BPclass))*100),"%")
        print("Classe >15mmHg: %d "%(self.BPclass[3]), "-> %.2f"%(self.BPclass[3]/(sum(self.BPclass))*100),"%")



# @brief:   This Function will make the SVM algorithm to estimate the value of SBP and DBP
# @param:   data is a dataframe with the features x samples
# @param:   features is a list of labels from data 
# @return:  void
class SupportVectorRegression():
    def __init__(self, data, features, label):
        self.x = data[[*features]]  # Features
        self.y = data[label]  # labels
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
            erro += abs(yPred[i] - yTest[i])
            desvio.append(abs(yPred[i] - yTest[i]))
            self.classify_BP(abs(yPred[i] - yTest[i]))
            #time.sleep(1)
        print(" ------- END OF ANALYSIS ------")
        print("N° de testes: ", yPred.shape[0])
        print("Mean Error: %.3f" %(erro/yPred.shape[0]))
        print("Standar Deviation NP: %.3f"%(np.std(desvio)))
        print("Total: %d" %(sum(self.BPclass)))
        print("Classe 5mmHg: %d "%(self.BPclass[0]), "-> %.2f"%(self.BPclass[0]/(sum(self.BPclass))*100),"%")
        print("Classe 10mmHg: %d "%(self.BPclass[1]), "-> %.2f"%(self.BPclass[1]/(sum(self.BPclass))*100), " % -",((self.BPclass[0]+self.BPclass[1])/(sum(self.BPclass))*100),"%")
        print("Classe 15mmHg: %d "%(self.BPclass[2]), "-> %.2f"%(self.BPclass[2]/(sum(self.BPclass))*100), " % -",((self.BPclass[0]+self.BPclass[1]+self.BPclass[2])/(sum(self.BPclass))*100),"%")
        print("Classe >15mmHg: %d "%(self.BPclass[3]), "-> %.2f"%(self.BPclass[3]/(sum(self.BPclass))*100),"%")



# @brief:   This Function will get the data separeted from one case of queensland dataset
# @param:   case str wich one shows the registered case number to be used
# @return:  return three list with the data from PPG and the references from SBP and DBP 
def getDataSepareted(case):
    pletData1,pletData2, SBPData, DBPData, Data = [], [], [], [], []
    path = "../../queensland_dataset/"+case+"/fulldata/"
    filemName = "uq_vsd_"+case+"_fulldata_13.csv"
    # Open csv file and get data
    c = 0
    with open(path+filemName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pletData2.append(float(row['Pleth']))
            if(( c < 13500 or c > 17500) and (c<31500 or c > 35500) and (c<49000 or c > 54000)):
            #if((c > 6000) and (c<19000 or c > 24000) and (c<37500 or c > 42000) and (c < 55000)):
                pletData1.append(float(row['Pleth']))
                SBPData.append((row['NBP (Sys)']))
                DBPData.append((row['NBP (Dia)']))
                Data.append([ (row['Pleth']), (row['NBP (Sys)']), (row['NBP (Dia)']) ])
                                  
            c+=1
    path = "../../myQueData/"+case+"/"
    filemName = "uq_"+case+"_12.csv"
    #saveSignal(path, filemName, Data)
    return pletData1,pletData2, SBPData, DBPData
 #end 

def saveSignal(path, filemName, data):
    dataFile = open(path+filemName, "w")
    dataFile.write("PPG, SBP, DBP \n")
    for x in data:
        dataFile.write( x[0]+"," + \
                        x[1]+"," + \
                        x[2]+"\n"
                        )
    print("Sinal salvo:", filemName)
    dataFile.close()
#End

# @brief:   This Function will get the entire data from one case of queensland dataset
# @param:   case str wich one shows the registered case number to be used
# @return:  return three list with the data from PPG and the references from SBP and DBP 
def getDataN(case):
    pletData, SBPData, DBPData = [], [], []
    path = "../../myQueData/"+case+"/"
    sample = 2
    filemName = "uq_"+case
    # Open csv file and get data
    try:
        while True:
            if(sample<10):
                filemName  = "uq_"+case+"_0"+str(sample)+".csv"
            else:
                filemName  = "uq_"+case+"_"+str(sample)+".csv"
            with open(path+filemName) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        if( ( float(row[' SBP'])> 70 ) and ( float(row[' DBP ']) > 40 ) ):
                            pletData.append(float(row['PPG']))
                            SBPData.append(float(row[' SBP']))
                            DBPData.append(float(row[' DBP ']))
                    except:
                        #print("Expept")
                        pass
            sample+=1
    except Exception as err:  # não pega MemoryError, SystemExit, KeyboardInterrupt
        print("Handling run-time error:",err)
        raise # sempre suba novamente a exceção não tratada, 
             # para que seja visível
    finally:
        return pletData, SBPData, DBPData


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

#PPG_Raw,PPG_full, SBPr, DBPr = getDataSepareted(case)
PPG_Raw, SBPr, DBPr = getDataN(case)
sps = 100

# Filtering signals
#PPG
lowcut = 0.5 
highcut = 8
order = 2 
PPGf = butter_bandpass_filter_zi(PPG_Raw, lowcut, highcut, sps, order)

window = slidingWindow(PPG_Raw,SBPr,DBPr)
window.computeWindow()

ft = window.featuresName
ft.remove('SBP')
ft.remove('DBP')
SBP_MLR = MultipleLinearRegression(window.dataFrame, ft, 'SBP')
DBP_MLR = MultipleLinearRegression(window.dataFrame,ft,'DBP')
SBP_SVR = SupportVectorRegression(window.dataFrame,ft,'SBP')
DBP_SVR = SupportVectorRegression(window.dataFrame,ft,'DBP')
#
SBP_MLR.compute(0.3)
SBP_SVR.compute(0.3)
#
DBP_MLR.compute(0.3)
DBP_SVR.compute(0.3)
##Make graph
#doGraphics(sps, PPG_Raw, PPGf)

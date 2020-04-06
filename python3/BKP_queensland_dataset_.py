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
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import csv
from processing.readandfilter import *
from processing.mathprocessing import *
from processing.makegraphics import *

# @brief:   This Function make graphics from PPG and ECG signals
# @param:   
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
def getD3(case):
    pletData = []
    path = "../../queensland_dataset/"+case+"/fulldata/"
    filemName = "uq_vsd_"+case+"_fulldata_01.csv"
    # Open csv file and get data
    with open(path+filemName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pletData.append(float(row['Pleth']))
    return pletData

def getD5(case):
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

# Read dataset
def getData(case):
    pletData = []
    path = "../../queensland_dataset/"+case+"/fulldata/"
    filemName = "uq_vsd_"+case+"_fulldata_01.csv"
    # Open csv file and get data
    with open(path+filemName) as dataFile:
        next(dataFile)
        for line in dataFile:
            aux = line.split(",")
            #print(aux[52])
            pletData.append(float(aux['52']))
        #end-for
    #end-with
    dataFile.close()
    return pletData
#end def

def getD(case):
    # Lendo o csv 
    pletData = []
    path = "../../queensland_dataset/"+case+"/fulldata/"
    sample = 2
    filemName = "uq_vsd_"+case+"_fulldata_"
    # Open csv file and get data
    try:
        if(sample<10):
            aux = pd.read_csv(path+filemName+"0"+str(sample)+".csv", usecols=['Pleth'])
        else:
            aux = pd.read_csv(path+filemName+str(sample)+".csv", usecols=['Pleth'])
        pletData.append(aux['Pleth'])
        
        sample+=1
    except:
        print("File doesn't exist")
    return aux
        
def getDa(case):
    # Lendo o csv 
    pletData = []
    path = "../../queensland_dataset/"+case+"/fulldata/"
    sample = 1
    filemName = "uq_vsd_"+case+"_fulldata_"
    # Open csv file and get data
    try:
        while True:
            if(sample<10):
                aux = pd.read_csv(path+filemName+"0"+str(sample)+".csv", usecols=['Pleth'])
            else:
                aux = pd.read_csv(path+filemName+str(sample)+".csv", usecols=['Pleth'])
            pletData.append(aux['Pleth'])
            
            sample+=1
    except:
        print("File doesn't exist")
    return pletData
        
    
#end def


# ----------------------------------------- MAIN ----------------------------------------------------------------------------------
case = str(input("Case (01:32): "))
case = "case"+case
#sampleData = str(input("Sample (1:13): "))

PPG_Raw = []
#PPG_Raw = getData(case)
PPG_Raw, SBPr, DBPr = getD5(case)
sps = 100
#print(PPG_Raw)
#print(len(PPG_Raw))
# Filtering signals
#PPG
lowcut = 0.5 
highcut = 8
order = 2 
PPGf = butter_bandpass_filter_zi(PPG_Raw, lowcut, highcut, sps, order)
#PPGf = PPG_Raw

#Make graph
doGraphics(sps, PPG_Raw, PPGf)

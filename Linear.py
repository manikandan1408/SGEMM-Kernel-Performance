#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sympy as sym
import numpy as np
import random 
import statistics as st
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random as rd
from sklearn.model_selection import train_test_split


# In[2]:


#Loading Dataset
os.chdir('C:\\Users\manit\OneDrive\Desktop\Masters\Spring20\Sabir')
dataFile = pd.read_csv("sgemm_product.csv")


# In[3]:


#Target Variable
pd.DataFrame.rename(dataFile,columns={'Run1 (ms)':'Run1','Run2 (ms)':'Run2','Run3 (ms)':'Run3', 'Run4 (ms)':'Run4'},inplace =True)
dataFile['Runs']=dataFile.apply(lambda row:(row.Run1+row.Run2+row.Run3+row.Run4)/4,axis=1)
dataFile = dataFile.drop(["Run1","Run2","Run3","Run4"], axis=1)


# In[4]:


#Data Pre-processing
columnList = list(dataFile.columns)
for column in dataFile.columns.difference(['STRM', 'STRN', 'SA', 'SB']):
     dataFile[column] = (dataFile[column]-st.mean(dataFile[column]))/(st.stdev(dataFile[column]))
train_data, test_data = train_test_split(dataFile, test_size = 0.3, random_state = 1234,shuffle = True)


# In[35]:


#Correlation

corr = dataFile.corr()
corr.style.background_gradient(cmap='YlGnBu')


# In[37]:


corr.to_csv("C:\\New folder\corr2.csv")


# In[6]:


#Heatmap

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,cmap=sns.diverging_palette(20, 220, n=200),
    square=True)


# In[7]:


def createSymbols(n):
    n+=1
    return [sym.symbols('b0:%d'%n)]


# In[8]:


def CostFunction(y, x, b) :
    func = b[0]+ np.dot(b[1:],x)-y
    return np.sum(func**2)


# In[9]:


def diffCostFunction(sampleSize,y,x,b,indexOfParameter):
    J = 0
    if(indexOfParameter==0):
         J =b[0]+ np.dot(b[1:],x)-y
    else:
        J =np.dot((b[0]+ np.dot(b[1:],x)-y),  x.iloc[indexOfParameter-1])
    
    J = np.sum(J)
    return J*(1/(sampleSize))


# In[23]:


def LmDefined(noOfParameters, inputData, sampleSize, lRate, threshold, parametersInput,numberOfIterations):
    
    #Declaration
    newParameters={}
    cfValue=[]
    minAtIndex=-1
    
    #Creating Symbols
    createdSymbols=createSymbols(noOfParameters)
    b = list(createdSymbols[0])
    
    yData = inputData['Runs']
    xVal = (inputData.loc[:, inputData.columns != 'Runs']).transpose()
    parameters=dict(zip(b,parametersInput))
    
    #Initialise Cost Function
    cfValue.append(CostFunction(yData , xVal, parametersInput))
    
    for index in range(numberOfIterations) :
        for parameter in parameters :
            newParameters[parameter]=parameters[parameter]-lRate*(diffCostFunction(sampleSize, yData, xVal, list(parameters.values()),list(parameters.keys()).index(parameter)))
        
        newCfValue= CostFunction(yData, xVal, list(newParameters.values()))
        if abs(newCfValue-cfValue[-1]) < threshold :
            if minAtIndex==-1:
                minAtIndex=index   
    
        parameters.update(newParameters)
        cfValue.append(newCfValue)
    if minAtIndex!=-1 :
        print ("min is achieved at "+str(minAtIndex) + " iteration and the squared error is "+str(cfValue[minAtIndex]))
    else:
        print ("squared error is not converging with the given parameters")
    return [cfValue, minAtIndex]


# In[24]:


#LearningRate=0.25,Threshold=0.00001


# In[26]:


parametersInput = [0.21, 0.25, 0.19, 0.20, 0.22, 0.24, 0.26, 0.18, 0.27, 0.28, 0.29,0.31,0.18, 0.32, 0.20]
plt.plot(LmDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.25,0.00001,parametersInput,700)[0],label="Train")
plt.plot(LmDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.25,0.00001,parametersInput,700)[0],label="Test" )


# In[ ]:


#Experiment1 with Learning rate= 0.1,0.2,0.3 & Threshold=0.0001 & No. of Iterations=700


# In[14]:


parametersInput = [0.21, 0.25, 0.19, 0.20, 0.22, 0.24, 0.26, 0.18, 0.27, 0.28, 0.29,0.31,0.18, 0.32, 0.20]
plt.ylim(100000, 110000)
plt.xlim(0,400)
plt.title("Cost function pattern of training data (Linear)")
plt.ylabel("CostFunction Value")
plt.xlabel("No. of Iterations")
plt.plot(LmDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.10,0.0001,parametersInput,700)[0],label="LearningRate: " +str(0.10))
plt.plot(LmDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.20,0.0001,parametersInput,700)[0],label="LearningRate: " +str(0.20))
plt.plot(LmDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.30,0.0001,parametersInput,700)[0],label="LearningRate: " +str(0.30))
plt.legend()


# In[15]:


plt.title("Cost function pattern of test data (Linear)")
plt.ylabel("CostFunction Value")
plt.xlabel("No. of Iterations")
plt.ylim(40000 , 50000)
plt.xlim(0,400)
plt.plot(LmDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.10,0.0001,parametersInput,700)[0],label="LearningRate: " +str(0.10))
plt.plot(LmDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.20,0.0001,parametersInput,700)[0],label="LearningRate: " +str(0.20))
plt.plot(LmDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.30,0.0001,parametersInput,700)[0],label="LearningRate: " +str(0.30))
plt.legend()


# In[16]:


#Experiment2 with threshold= 0.0001,0.001,0.01
parametersInput = [0.21, 0.25, 0.19, 0.20, 0.22, 0.24, 0.26, 0.18, 0.27, 0.28, 0.29,0.31,0.18, 0.32, 0.20]
plt.title("Cost function pattern of train data (Linear)")
plt.ylabel("CostFunction Value")
plt.xlabel("No. of Iterations")
plt.ylim(0 ,150000)
thresholds = [0.0001,0.001, 0.01]
plt.xlim(0, 300)
for threshold in thresholds:
    result = LmDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.30, threshold, parametersInput,500)
    for index in range(result[1],500):
        result[0][index] = 0
    plt.plot(result[0],label="Threshold:" +str(threshold))
    plt.legend()


# In[17]:


plt.title("Cost function pattern of test data (Linear)")
plt.ylabel("CostFunction Value")
plt.xlabel("No. of Iterations")
plt.ylim(0 ,150000)
thresholds = [0.0001, 0.001, 0.01]
plt.xlim(0, 300)
for threshold in thresholds:
    result = LmDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.30, threshold, parametersInput,500)
    for index in range(result[1],500):
        result[0][index] = 0
    plt.plot(result[0],label="Threshold:" +str(threshold))
plt.legend()


# In[18]:


parametersInput = [0.21, 0.25, 0.19, 0.20, 0.22, 0.24, 0.26, 0.18, 0.27, 0.28, 0.29,0.31,0.18, 0.32, 0.20]
plt.title("Cost function pattern for Threshold(Linear)")
plt.ylabel("CostFunction Value")
plt.xlabel("No. of Iterations")
plt.plot(LmDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.30,0.0001,parametersInput,500)[0],label="Train")
plt.plot(LmDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.30,0.0001,parametersInput,500)[0],label="Test")


# In[19]:


plt.title("Cost function pattern of Complete data(Linear)")
plt.ylabel("CostFunction Value")
plt.xlabel("No. of Iterations")
plt.plot(LmDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.30,0.0001,parametersInput,500)[0],label="Train")
plt.plot(LmDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.30,0.0001,parametersInput,500)[0],label="Test")
plt.legend()


# In[22]:


import random as rd
random.seed(15432)
colIndices = list(rd.sample(range(14),10))
colIndices.sort()
colIndices
colList = dataFile.columns
colList = list(colList[colIndices])
print(colList)
colList.append("Runs")
parametersInputNew = [0.21, 0.25, 0.19, 0.20, 0.22, 0.24, 0.26, 0.18, 0.27, 0.28, 0.29]
plt.title("Cost function pattern of data with 10 random columns(Linear)")
plt.ylabel("CostFunction Value")
plt.xlabel("No. of Iterations")
plt.plot(LmDefined(len(colList)-1, train_data[colList], len(train_data) ,0.30,0.0001,parametersInputNew,500)[0],label="Train")
plt.plot(LmDefined(len(colList)-1, test_data[colList], len(test_data) ,0.30,0.0001,parametersInputNew,500)[0],label="Test")
plt.legend()


# In[21]:


excludeColumns=['KWG', 'MDIMA','NDIMB', 'STRN']
selectedColumns = list(dataFile.columns)
for feature in excludeColumns:
    selectedColumns.remove(feature)
print(selectedColumns)
parametersInputNew = [0.21, 0.25, 0.19, 0.20, 0.22, 0.24, 0.26, 0.18, 0.27, 0.28, 0.29]
plt.title("Cost function pattern of data with 10 selected columns(Linear)")
plt.ylabel("CostFunction Value")
plt.xlabel("No. of Iterations")
plt.plot(LmDefined(len(selectedColumns)-1, train_data[selectedColumns], len(train_data) ,0.30,0.0001,parametersInputNew,500)[0],label="Train")
plt.plot(LmDefined(len(selectedColumns)-1, test_data[selectedColumns], len(test_data) ,0.30,0.0001,parametersInputNew,500)[0],label="Test")
plt.legend()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[5]:


#Loading Dataset
os.chdir('C:\\Users\manit\OneDrive\Desktop\Masters\Spring20\Sabir')
dataFile = pd.read_csv("sgemm_product.csv")


# In[6]:


#Target Variable
pd.DataFrame.rename(dataFile,columns={'Run1 (ms)':'Run1','Run2 (ms)':'Run2','Run3 (ms)':'Run3', 'Run4 (ms)':'Run4'},inplace =True)
dataFile['AverageRun']=dataFile.apply(lambda row:(row.Run1+row.Run2+row.Run3+row.Run4)/4,axis=1)
dataFile = dataFile.drop(["Run1","Run2","Run3","Run4"], axis=1)


# In[7]:


#Data Pre-processing
columnList = list(dataFile.columns)
for column in dataFile.columns.difference(['STRM', 'STRN', 'SA', 'SB']):
     dataFile[column] = (dataFile[column]-st.mean(dataFile[column]))/(st.stdev(dataFile[column]))
train_data, test_data = train_test_split(dataFile, test_size = 0.3, random_state = 1234,shuffle = True)


# In[68]:


#Correlation
corr = dataFile.corr()
corr.style.background_gradient(cmap='Greens')                        


# In[71]:


#Heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.title('Correlation between different features')


# In[8]:


def createSymbols(n):
    n+=1
    return [sym.symbols('b0:%d'%n)]


# In[9]:


def CostFunction(y, x, b) :
    func = b[0]+ np.dot(b[1:],x)-y
    return np.sum(func**2)


# In[10]:


def diffCostFunction(sampleSize,y,x,b,indexOfParameter):
    J = 0
    if(indexOfParameter==0):
         J =b[0]+ np.dot(b[1:],x)-y
    else:
        J =np.dot((b[0]+ np.dot(b[1:],x)-y),  x.iloc[indexOfParameter-1])
    
    J = np.sum(J)
    return J*(1/(sampleSize))


# In[11]:


def logitCostFunc(y, x ,b):
    
    func = np.reciprocal(1+np.exp((-1)*(b[0]+ np.dot(b[1:],x))))

    costFuncVal = -1* (np.sum(np.log(func[np.where(y==1)])) + np.sum(np.log(1-func[np.where(y==0)])))
    return costFuncVal/len(y)


# In[12]:


def sigmoidFuncVal(y ,x, b, indexOfParameter):
    func = np.reciprocal(1+np.exp((-1)*(b[0]+ np.dot(b[1:],x))))
    J = func -  y
    if(indexOfParameter != 0):
        J =np.dot(func - y,  x.iloc[indexOfParameter-1])
    J = np.sum(J)
    return (J/len(y))


# In[13]:


def logitDefined(noOfParameters, inputData, sampleSize, lRate, threshold, parametersInput,numberOfIterations):       
   
    newParameters={}
    cfValue=[]
    minAtIndex=-1
    
    createdSymbols=createSymbols(noOfParameters)
    b = list(createdSymbols[0])
    
    parameters=dict(zip(b,parametersInput))
    parameterKeysList=list(parameters.keys())
   
    medianVal = np.median(inputData['AverageRun'])
    y=np.array(inputData['AverageRun'])
    y = (y<medianVal).astype(int)
    x =(inputData.iloc[:,inputData.columns != 'AverageRun']).transpose()
    
    cfValue.append(logitCostFunc(y, x, list(parameters.values())))

    for index in range(1,numberOfIterations) :
        
        for parameter in parameters :
            indexVal= parameterKeysList.index(parameter)
            newParameters[parameter]=parameters[parameter]-lRate*(sigmoidFuncVal(y, x, list(parameters.values()), indexVal))
      
        newCfValue= logitCostFunc(y, x, list(newParameters.values()))
        if abs(newCfValue-cfValue[-1])<threshold :
            if minAtIndex==-1:
                minAtIndex=index
        
        parameters.update(newParameters)
        cfValue.append(newCfValue)
        
    if minAtIndex!=-1 :
        print ("min is achieved at "+str(minAtIndex) + " iteration and the squared error is "+str(cfValue[minAtIndex]))
    else:
        print ("squared error is not converging with the given parameters")
    return [cfValue, minAtIndex]


# In[14]:


#Experiment1 - Error/Accuracy Variation for training sets
parametersInput = [0.21, 0.25, 0.19, 0.20, 0.22, 0.24, 0.26, 0.18, 0.27, 0.28, 0.29,0.31,0.18, 0.32, 0.20]
plt.title("Cost function pattern of training data (Logistic)")
plt.ylabel("CostFunction Value")
plt.xlabel("No. of Iterations")
plt.ylim(0.40 , 0.65)
plt.xlim(0,300)
plt.plot(logitDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.10,0.0001,parametersInput,700)[0],label="LearningRate: " +str(0.10))
plt.plot(logitDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.20,0.0001,parametersInput,700)[0],label="LearningRate: " +str(0.20))
plt.plot(logitDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.30,0.0001,parametersInput,700)[0],label="LearningRate: " +str(0.30))
plt.legend()


# In[17]:


plt.title("Cost function pattern of test data (Logistic)")
plt.ylabel("CostFunction Value")
plt.xlabel("No. of Iterations")
plt.ylim(0.40 , 0.65)
plt.xlim(0,300)
plt.plot(logitDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.10,0.0001,parametersInput,700)[0],label="LearningRate: " +str(0.10))
plt.plot(logitDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.20,0.0001,parametersInput,700)[0],label="LearningRate: " +str(0.20))
plt.plot(logitDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.30,0.0001,parametersInput,700)[0],label="LearningRate: " +str(0.30))
plt.legend()


# In[18]:


#Experiment2- Changing the threshold values 
plt.title("Cost function pattern of train data (Logistic)")
plt.ylabel("CostFunction Value")
plt.xlabel("No. of Iterations")
plt.xlim (0, 500)
thresholds = [0.0001, 0.001, 0.01]
for threshold in thresholds:
    result = logitDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.30,threshold,parametersInput,500)
    for index in range(result[1],500):
        result[0][index] = 0
    plt.plot(result[0],label="Threshold:" +str(threshold))

plt.legend()


# In[80]:


parametersInput = [0.26, 0.18, 0.27, 0.28, 0.29,0.31,0.18, 0.32, 0.21, 0.25, 0.19, 0.20, 0.22, 0.24, 0.20]
plt.title("Logistic Cost function(test data)")
plt.ylabel("Cost Value")
plt.xlabel("# of Iterations")
plt.xlim (0, 500)
thresholds = [0.0001, 0.001, 0.01]
for threshold in thresholds:
    result = logitDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.30,threshold,parametersInput,500)
    for index in range(result[1],500):
        result[0][index] = 0
    plt.plot(result[0],label="Threshold:" +str(threshold))

plt.legend()


# In[81]:


parametersInput = [0.26, 0.18, 0.27, 0.28, 0.29,0.31,0.18, 0.32, 0.21, 0.25, 0.19, 0.20, 0.22, 0.24, 0.20]
plt.title("Logistic Cost function trend for Threshold")
plt.ylabel("Cost Value")
plt.xlabel("# of Iterations")
plt.ylim(0.40,0.70)
plt.plot(logitDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.30,0.0001,parametersInput,500)[0],label="Train")
plt.plot(logitDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.30,0.0001,parametersInput,500)[0],label="Test")


# In[82]:


#Experiment3 - Retrain on ten random features
random.seed(15432)
colIndices = list(rd.sample(range(14),10))
colIndices.sort()
colIndices
colList = dataFile.columns
colList = list(colList[colIndices])
print(colList)
colList.append("AverageRun")
parametersInputNew = [0.26, 0.18, 0.27, 0.28, 0.29,0.31,0.18, 0.32, 0.21, 0.25, 0.19]
plt.title("Cost function trend for data with 10 random Features(Logit)")
plt.ylabel("Cost Value")
plt.xlabel("# of Iterations")
plt.ylim(0.40,0.60)
plt.plot(logitDefined(len(colList)-1, train_data[colList], len(train_data) ,0.30,0.0001,parametersInputNew,300)[0],label="Train")
plt.plot(logitDefined(len(colList)-1, test_data[colList], len(test_data) ,0.30,0.0001,parametersInputNew,300)[0],label="Test")
plt.legend()


# In[83]:


plt.title("Cost function pattern with all the features(Logit)")
plt.ylabel("Cost Value")
plt.xlabel("# of Iterations")
plt.ylim(0.40,0.60)
plt.plot(logitDefined(len(columnList)-1, train_data[columnList], len(train_data) ,0.30,0.0001,parametersInput,300)[0],label="Train")
plt.plot(logitDefined(len(columnList)-1, test_data[columnList], len(test_data) ,0.30,0.0001,parametersInput,300)[0],label="Test")
plt.legend()


# In[84]:


#Experiment4
excludeColumns=['KWG', 'MDIMA','NDIMB', 'STRM']
selectedColumns = list(dataFile.columns)
for feature in excludeColumns:
    selectedColumns.remove(feature)
print(selectedColumns)
parametersInputNew = [0.26, 0.18, 0.27, 0.28, 0.29,0.31,0.18, 0.32, 0.21, 0.25, 0.19]
plt.title("Cost function trend with 10 columns(Logistic)")
plt.ylabel("Cost Value")
plt.xlabel("# of Iterations")
plt.ylim(0.40,0.60)
plt.plot(logitDefined(len(selectedColumns)-1, train_data[selectedColumns], len(train_data) ,0.30,0.0001,parametersInputNew,500)[0],label="Train")
plt.plot(logitDefined(len(selectedColumns)-1, test_data[selectedColumns], len(test_data) ,0.30,0.0001,parametersInputNew,500)[0],label="Test")
plt.legend()


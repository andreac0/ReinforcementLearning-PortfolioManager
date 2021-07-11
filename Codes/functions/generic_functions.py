
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import numpy as np
import math
import tensorflow as tf
import pandas as pd
import copy
import random
from numpy import array
from sklearn.decomposition import PCA
from sklearn import preprocessing
from tensorflow.keras import Sequential
from tensorflow.random import set_seed
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
import seaborn as sns
from tensorflow.keras.optimizers import Adam
import statistics 
from math import sqrt
from numpy import mean
from scipy.stats import sem
from scipy.stats import t
import yfinance as yf
from datetime import datetime
from tensorflow import keras
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.decomposition import KernelPCA
###################################
# Standardization on train set and then applied to test set and to the commission
###################################
def standardize(data,index, commission=0.025, ret=False):
    md,std=data[:index].mean(axis=0),data[:index].std(axis=0)
    if ret: return((data-md)/std,(commission-md)/std)
    else: return((data-md)/std)

###################################
# Normalization
###################################
def normalize(data,index):
  mini=data[:index].max()
  maxi=data[:index].min()
  y=(data-mini)/(maxi-mini)
  return(y)


###################################
# Kernel PCA: computed on train set and extended on the test set
###################################
def kernel_pca(data, index, num_components=2):
    kpca = KernelPCA(n_components=num_components, kernel='rbf')
    k_component_train=kpca.fit_transform(data[:index,:])
    k_component_test=kpca.transform(data[index:,:])
    return(np.append(k_component_train,k_component_test,axis=0))    

###################################
# PCA semplice: computed on train set and extended on the test set
###################################
def my_pca(data, index, num=2):
    pca = PCA(n_components=num)
    principal_component=pca.fit_transform(data[:index,:])
    pca_test=pca.transform(data[index:,:])
    return(np.append(principal_component,pca_test,axis=0))


###################################
# Function to fix dates on csv file used 
###################################
def allinate_dates(X,data):
   y=data.index
   X=array(X)
   date=str()
   for i in range(len(X)):
     date=np.append(date,datetime.strptime(str(int(X[i,0])), '%Y%m%d').strftime('%Y-%m-%d'))
   date=date[1:]
   date=pd.DatetimeIndex(date)
   n=np.in1d(date,y)   
   for j in range(len(X)):
       if n[j]==True:
           k=j
           break
   X=X[k:,:]
   date=str()
   for i in range(len(X)):
     date=np.append(date,datetime.strptime(str(int(X[i,0])), '%Y%m%d').strftime('%Y-%m-%d'))
   date=date[1:]
   date=pd.DatetimeIndex(date)
   n=np.in1d(date,y) 
   for j in range(len(X)):
       if n[j]==False:
           k=j
           break
   X=X[:k,:]
   
   date=str()
   for i in range(len(X)):
     date=np.append(date,datetime.strptime(str(int(X[i,0])), '%Y%m%d').strftime('%Y-%m-%d'))
   date=date[1:]
   date=pd.DatetimeIndex(date)
   
   n=np.in1d(y,date)    
   newX=X[0,1:].reshape(-1,2)
   c=1
   for i in range(1,len(n)):
       if n[i]==True:
           newX=np.append(newX,X[c,1:].reshape(-1,2),axis=0)
           c=c+1
       else:
           newX=np.append(newX,newX[-1,:].reshape(-1,2),axis=0)   
   return(newX)

 #####################
 # Test on results
 #####################
def independent_ttest(data1, data2):
	# calculate means
    mean1,mean2=mean(data1),mean(data2)
	# calculate standard errors
    se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
    sed = sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
    t_stat = (mean1 - mean2) / sed
	# degrees of freedom
    df = len(data1) + len(data2) - 2
	# calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0

    return  mean1,mean2, round(p,3)

def tests(misure):
 l=misure[0][1].index
 k=array([]).reshape(-1,3)
 media_nulla=array([])
 for t in l:
   x,y=array([]),array([])
   for i in range(len(misure)):
      x=np.append(x,misure[i][0][t])
      y=np.append(y,misure[i][1][t])
   k=np.append(k,array(independent_ttest(x,y)).reshape(-1,3),axis=0) 
   media_nulla=np.append(media_nulla,round(scipy.stats.ttest_1samp(x,0)[1],3))
 k=pd.DataFrame(np.append(k,media_nulla.reshape(-1,1),axis=1),columns=("Media RL","Media B&H","P-value diff Medie","P-value media nulla"),index=l)  
 return(k)

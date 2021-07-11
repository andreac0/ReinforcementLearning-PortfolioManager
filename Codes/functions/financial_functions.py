# Definition of financial functions used
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
# Sharpe Ratio
###################################
def sharpe_ratio(series,rf):
  '''
  Returns the annual Sharpe Ratio

  Parameters:
      series: Pandas Series of financial returns
      rf: risk-free rate
  '''
  return((series.mean()-rf)*np.sqrt(252)/series.std())

###################################
# Sortino Ratio
###################################
def sortino(returns, index, rf=0.01/100, complete=True):
  '''
  Returns the annual Sortino Ratio

  Parameters:
      series: Pandas Series of financial returns
      index: position of the time series that splits train and test
      rf: risk-free rate  
      complete: boolean -> if true compute the sortino rate for the entire series and only for the test part
  '''
  DSR=returns[(returns<rf)].std()
  S_tot=np.sqrt(252)*(returns.mean()-rf)/DSR
    
  if complete:
     returns=returns[index:]
     DSR_test=returns[(returns<rf)].std()
     S_test=np.sqrt(252)*(returns.mean()-rf)/DSR
     return((S_tot,S_test))
  else:
     return((S_tot))

###################################
# Maximum Drawdown
###################################
def MDD(returns,index, complete=True):
  '''
  Returns the Maximum Drawdown

  Parameters:
    returns: Pandas Series of financial returns
    index: position of the time series that splits train and test
    complete: boolean -> if true compute the sortino rate for the entire series and only for the test part
  '''
  returns=pd.Series(returns)
  cum_rets = (1 + returns).cumprod() - 1
  nav = ((1 + cum_rets) * 100).fillna(100)
  hwm = nav.cummax()
  dd = nav / hwm - 1
    
  if complete:
    cum_rets_test=(1 + returns[index:]).cumprod() - 1
    nav_test = ((1 + cum_rets_test) * 100).fillna(100)
    hwm_test = nav_test.cummax()
    dd_test = nav_test / hwm_test - 1
    return(abs(min(dd)),abs(min(dd_test)))

  else: return((abs(min(dd))))

###################################
# Softmax function to build portfolio shares
###################################
def softmax(x):
  '''
  Softmax function that returns a vector with portfolio shares 

  Parameters: 
     x: vector
  '''
  return(np.exp(x)/np.sum(np.exp(x)))


###################################
# RSI index function
###################################
def computeRSI (data, time_window):
  '''
  Compute the RSI index 

  Parameters: 
    data: Pandas series of financial returns
    time_window: window to use to compute the RSI
  '''
  diff = data.diff(1).dropna()        # diff in one field(one day)
  #this preservers dimensions off diff values
  up_chg = 0 * diff
  down_chg = 0 * diff
  # up change is equal to the positive difference, otherwise equal to zero
  up_chg[diff > 0] = diff[ diff>0 ]
  # down change is equal to negative deifference, otherwise equal to zero
  down_chg[diff < 0] = diff[ diff < 0 ]
  up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
  down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
  rs = abs(up_chg_avg/down_chg_avg)
  rsi = 100 - 100/(1+rs)

  return rsi


#####################
# Data download function from Yahoo finance
#####################


def download(name, window = 14, test_years = 1):
  '''
  Input: stock tickers of the companies to be downloaded

  Output: 1. Adjusted close price time series 
          2. Returns time series 
          3. DataFrame object with all the variables that defines the state, allinated to the dates. Kernel PCA has been used to reduce number of attributes
  '''

  p_close, others, returns = list(), list(), list() 

  for i in range(len(name)):
     # Initial download of the data, in order to compute the RSI
   start_date=pd.to_datetime("2012-01-01")
   end_date=pd.to_datetime("2020-01-01")
   data=yf.download(name[i], start=start_date, end=end_date)
  
    # RSI index  
   rsi=computeRSI(data["Adj Close"], window)
   start_date=rsi.index[window-1]

    # Other useful indexes
   other_priceindex=array(np.log((data/data.shift(1))[start_date:]))[:,:-1]

   # Download of the data, in the new window
   data=yf.download(name[i], start=start_date, end=end_date)[start_date:]
   giorno=data.index.day_name()
   index=len(data)-252*test_years
  
   # Treasury bills
   treasury_bills=pd.read_csv("~/bill_rates1yearPY.csv",names=["date","short_term_bill"],parse_dates=["date"],index_col=0)
   dates_to_consider=np.in1d(treasury_bills.index,data.index)
   treasury_bills=standardize(array(treasury_bills[dates_to_consider]),index).reshape(-1,1)

   # Risk Aversion index
   risk_aversion=pd.read_csv("~/index.csv",index_col=0)
   risk_aversion=standardize(allinate_dates(risk_aversion, data),index)
  
   # Compute of returns
   ret=array(np.log(data["Adj Close"]/data.shift(1)["Adj Close"]))[1:]
   returns.append(ret)
   
   # Trading Volume
   volume=standardize(array(data)[:,-1], index).reshape(-1,1)
    
   # Union attributes
   other=np.concatenate((treasury_bills, risk_aversion, other_priceindex,volume),axis=1)
   other=np.append(other, array(rsi[start_date:]/100).reshape(-1,1),axis=1)

   # Implied volatility
   implied_volatility = array(yf.download('^vix', start=start_date, end=end_date)[start_date:])[:,4].reshape(-1,1)
   other=np.append(other,standardize(implied_volatility, index),axis=1)

   # S&P500
   standard_poors = standardize(array(yf.download('^gspc', start=start_date, end=end_date)[start_date:]),index)
   principal_component=kernel_pca(standard_poors, index)
   other=np.append(other,principal_component,axis=1)

   # Dow jones
   dowjones = standardize(array(yf.download('^dji', start=start_date, end=end_date)[start_date:]),index)
   principal_component=kernel_pca(dowjones, index)
   other=np.append(other,principal_component,axis=1)
  
   # Nasdaq
   nasdaq = standardize(array(yf.download('^ixic', start=start_date, end=end_date)[start_date:]),index)
   principal_component=kernel_pca(nasdaq, index)
   other=np.append(other,principal_component,axis=1)
  
   # Name of the day
   giorno=array(giorno).reshape(-1,1)
   labels = ('Monday', 'Tuesday', 'Wednesday','Thursday','Friday')
   for i in range(len(labels)):
     giorno[np.where(giorno==labels[i])] = i+1
   other=np.append(other,array(giorno, dtype=int),axis=1)
  
   p_close.append(array(data["Adj Close"]))
  
   other=other[1:,]
   others.append(other)
  
  return(p_close,returns,others)

# Compute results on estimated model
def model_results(ret, model,back,actions,test_index,name,raw_price,other_data=array([]),commission=0.0025,graph=True):
    ret2=copy.deepcopy(ret)
    ret=standardize(ret, test_index)
    azione=np.repeat(array([0]),back) 
    num_input=(back+other_data.shape[1]+1)
    weights=np.repeat(0,back)

    for i in range(back,len(ret)):
        q_value=model.predict(np.append(ret[i-back:i],np.append(other_data[i-1,:],azione[-1])).reshape(-1,num_input))
        azione=np.append(azione,actions[np.argmax(q_value)])
        weights=np.append(weights,np.amax(q_value))
       
    returns=array([])
    for i in range(len(azione)):
        returns=np.append(returns,azione[i]*ret2[i]-commission*(azione[i]!=azione[i-1]))
    cumreturns=returns.cumsum()
    buy_hold=ret2.cumsum()
    
    complete_series=np.concatenate((raw_price[1:].reshape(-1,1),ret2.reshape(-1,1),azione.reshape(-1,1),cumreturns.reshape(-1,1),weights.reshape(-1,1)),axis=1)

    if graph:
      print(cumreturns[-1])
      fig=plt.figure(1,figsize=(12, 8))
      fig.suptitle('Deep Q-Network',fontsize=30)
      fig.add_subplot(411)
      plt.plot(raw_price)
      plt.legend([name])
      fig.add_subplot(412)
      plt.plot(ret2)
      fig.add_subplot(413)
      plt.plot(azione)
      fig.add_subplot(414)
      plt.plot(cumreturns)
      plt.plot(cumreturns[:test_index])
      plt.plot(buy_hold)
      plt.legend(["Test","Train","Hold"])
      
      test_cum=np.cumsum(returns[test_index:])
      print("test result:",test_cum[-1],'\n',"B&H:",buy_hold[-1],'\n','B&H test:',ret2[test_index:].cumsum()[-1])
    return(complete_series)

# Strategy comparison
def misure_val(data,index,name,rf=0.01/100, commission=0.0025):
    print(name)
    my_strategy=array([])
    for i in range(len(data)):
        my_strategy=np.append(my_strategy,data[i,1]*data[i,2]-commission*(data[i,2]!=data[i-1,2]))
    cumreturns=my_strategy.cumsum()
    
    sr, sr_test=sharpe_ratio(my_strategy, rf),sharpe_ratio(my_strategy[index:], rf)
    volatility,vol_test=my_strategy.std(),my_strategy[index:].std()
    notlosing_train=np.sum((my_strategy[:index]>=0)*1)/((np.sum((my_strategy[:index]<0)*1))+np.sum((my_strategy[:index]>=0)*1))
    notlosing=np.sum((my_strategy[index:]>=0)*1)/((np.sum((my_strategy[index:]<0)*1))+np.sum((my_strategy[index:]>=0)*1))
    S,S_test=sortino(my_strategy,index)
    mdd,mdd_test=MDD(data[:,1]*data[:,2],index)
    print("RL strategy:","\n",
          "-Final result on complete serie:",round(cumreturns[-1],3),"\n",
          "-Final result only on test:",round(my_strategy[index:].cumsum()[-1],3),"\n",
          "-Sharpe ratios:",round(sr,3), "and on test", round(sr_test,3),"\n",
          "-Sortino ratio:",round(S,3),"and on test",round(S_test,3),"\n",
          "-MDD:",round(mdd,2),"and on test",round(mdd_test,2),"\n",
          "-Volatility %:",round(volatility,2), "and on test", round(vol_test,2),"\n",
          "-Correct actions train:",round(notlosing_train,2),"\n",
          "-Correct Actions on test:",round(notlosing,2))
    
    my=pd.Series((cumreturns[-1],my_strategy[index:].cumsum()[-1],sr,sr_test,S,S_test,mdd,mdd_test,volatility,vol_test,notlosing_train, notlosing),
                 index=("Final Res","Test Res","SR","SR Test","Sortino","Sortino test","MDD","MDD test","Volatilità", "Vol test","Correct actions train","Correct actions test"))
    
    plt.figure(1,figsize=(12, 8))
    plt.suptitle(name,fontsize=30)
    plt.plot(my_strategy[index:].cumsum())
    
    hold_strategy=data[:,1]
    holdcum=hold_strategy.cumsum()
    sr_hold,sr_hold_test=sharpe_ratio(hold_strategy, rf),sharpe_ratio(hold_strategy[index:], rf)
    volatility,vol_test=hold_strategy.std(),hold_strategy[index:].std()
    notlosing_train=np.sum((hold_strategy[:index]>=0)*1)/((np.sum((hold_strategy[:index]<0)*1))+np.sum((hold_strategy[:index]>=0)*1))
    notlosing=np.sum((hold_strategy[index:]>=0)*1)/((np.sum((hold_strategy[index:]<0)*1))+np.sum((hold_strategy[index:]>=0)*1))
    S,S_test=sortino(hold_strategy,index)
    mdd,mdd_test=MDD(hold_strategy,index)
    print("Hold stock strategy:","\n",
          "-Final result on complete serie:",round(holdcum[-1],3),"\n",
          "-Final result only on test:",round(hold_strategy[index:].cumsum()[-1],3),"\n",
          "-Sharpe ratios:",round(sr_hold,3), "and on test", round(sr_hold_test,3),"\n",
          "-Sortino ratio:",round(S,3),"and on test",round(S_test,3),"\n",
          "-MDD:",round(mdd,2),"and on test",round(mdd_test,2),"\n",
          "-Volatility %:",round(volatility,2), "and on test", round(vol_test,2),"\n",
          "-Correct actions train:",round(notlosing_train,2),"\n",
          "-Correct Actions on test:",round(notlosing,2))
    
    hold=pd.Series((holdcum[-1],hold_strategy[index:].cumsum()[-1],sr_hold,sr_hold_test,S,S_test,mdd,mdd_test,volatility,vol_test,notlosing_train,notlosing),
                 index=("Final Res","Test Res","SR","SR Test","Sortino","Sortino test","MDD","MDD test","Volatilità", "Vol test","Correct actions train","Correct actions test"))

    plt.plot(hold_strategy[index:].cumsum())
   
    print("---------------------------------------------------------")
    
    plt.legend(["RL strategy","Hold"])
    plt.show()
    
    return(pd.concat([my,hold],axis=1))

###############################
# Build portfolio function
###############################

# Change portfolio weights
def new_port(action, weights, amount=1):
  '''
  Input: 
   - action: action to perform on a stock given by the model
   - weights: portfolio weights to be changed after the actions
   - amount: parameter that increase the entity of the actions
  '''
  sell=array(np.where(action==-1)).reshape(-1)
  buy=array(np.where(action==1)).reshape(-1)
  hold=array(np.where(action==0)).reshape(-1)
    
  weights[sell]=-amount*softmax(weights[sell])
  weights[buy]=amount*softmax(weights[buy])
  weights[hold]=np.repeat(0,len(hold))

  weights=softmax(weights)
    
  return(weights)

# Build and plot complete portofolio weights and returns through time

def build_portfolio(data, name, dates, commission_cost = 0.0025, entity=1, plot=True):
  '''
  Input:
   - data: original returns time series
   - name: names of companies
   - dates: vector of dates
   - commission_cost: % paid as transaction costs
   - entity: parameter to pass to the new_port function
   - plot: plot results
  '''
  size=len(data)
  num_obs=len(data[0])
  azioni=array([]).reshape(-1,num_obs)
  rendimenti=array([]).reshape(-1,num_obs)
  pesi_cambiamento=array([]).reshape(-1,num_obs)
  comm=array([]).reshape(-1,num_obs)
    
  for i in range(0,size):
        azioni=np.concatenate((azioni,data[i][:,2].reshape(-1,num_obs)))
        rendimenti=np.concatenate((rendimenti,data[i][:,1].reshape(-1,num_obs)))
        pesi_cambiamento=np.concatenate((pesi_cambiamento,data[i][:,4].reshape(-1,num_obs)))
        my_strategy=array([0])
        for k in range(1,num_obs):
          my_strategy=np.append(my_strategy,commission_cost*(data[i][k,2]!=data[i][k-1,2]))
        comm=np.concatenate((comm,my_strategy.reshape(-1,num_obs)))
        
  comm=np.transpose(comm)
  azioni=np.transpose(azioni)
  rendimenti=np.transpose(rendimenti)
  pesi_cambiamento=np.transpose(pesi_cambiamento)
    
  portfolio_weights=np.repeat(1/size,size).reshape(-1,size)
  value_portfolio=np.sum(portfolio_weights*rendimenti[0,:])
    
  equi_weighted=np.repeat(1/size,size).reshape(-1,size)
  equi_weighted=np.repeat(equi_weighted,num_obs,axis=0)
  value_equi_port=np.copy(value_portfolio)
    
  r=np.random.random(size)
  random_port=(r/np.sum(r)).reshape(-1,size)
  value_random=np.sum(random_port*rendimenti[0,:])
    
  for i in range(1,num_obs):
        new_weights=new_port(azioni[i,:], pesi_cambiamento[i,:],entity=entity)
        value_portfolio=np.append(value_portfolio,np.sum(new_weights*(rendimenti[i,:]-comm[i,:]/size)))
        portfolio_weights=np.concatenate((portfolio_weights,new_weights.reshape(-1,size)),axis=0)
        
        value_equi_port=np.append(value_equi_port,np.sum(equi_weighted[i,:]*rendimenti[i,:]))
        
        random_port=np.append(random_port,(r/np.sum(r)).reshape(-1,size),axis=0)
        value_random=np.append(value_random,np.sum(random_port[i,:]*rendimenti[i,:]))
      
  cum_equi_value=np.cumsum(value_equi_port)
  cum_value=np.cumsum(value_portfolio)
  cum_random=np.cumsum(value_random)
  
  if plot:
      y = np.row_stack((np.transpose(portfolio_weights)))
      x = np.arange(len(portfolio_weights))
      percent = y/y.sum(axis=0).astype(float) * 100 

      fig=plt.figure(figsize=(12, 7))
      plt.plot(pd.Series(cum_value,index=dates))
      plt.plot(pd.Series(cum_equi_value,index=dates),color="green")
      #plt.plot(cum_random)
      plt.legend(["RL Portfolio","Portfolio Equidistribuito","Random"])
      plt.margins(0, 0)
      plt.show() 
      fig=plt.figure(figsize=(12, 7))
      plt.stackplot(dates, percent) 
      plt.legend(name)
      plt.margins(0, 0) 
      plt.show() 
 
  return(list((portfolio_weights,cum_value,cum_equi_value,cum_random,rendimenti,value_portfolio,comm)))

 # Test if the defined rule works
def portfolio_method(risultati,name,commission,entity,dates):
   portfolio_weights,My_port,Equi_port,Random_port, Rendimenti,myret,c=build_portfolio(risultati,name,dates,commission,entity, plot=False)    
   
   risultati2=copy.deepcopy(risultati)
   
   for i in range(len(risultati2)):
      risultati2[i][np.where(risultati2[i][:,2]==1),-1]=1
      risultati2[i][np.where(risultati2[i][:,2]==0),-1]=0
      risultati2[i][np.where(risultati2[i][:,2]==-1),-1]=-1
   
   portfolio_weights2,My_port2,Equi_port2,Random_port2, Rendimenti2,myret,c=build_portfolio(risultati2,name,dates,commission,entity, plot=False)

   fig=plt.figure(figsize=(10, 7))
   plt.plot(pd.Series(My_port,index=dates),color="purple")
   plt.plot(pd.Series(My_port2,index=dates), color="green")
   plt.legend(["Regola dei values","Valori Arbitrari"])
   plt.show()
   fig=plt.figure(figsize=(10, 7))
   plt.margins(0, 0)
   plt.plot(pd.Series(My_port-My_port2,index=dates))
   plt.show()

# Portfolio comparison
def valuta_port(pesi, rendimenti,my_rendimenti,rf=0.01/100):
    index=len(pesi)-252
    num_title=pesi.shape[1]
    mySR=sharpe_ratio(my_rendimenti, rf)
    mysortino=sortino(my_rendimenti, index,rf,complete=False)
    my_mdd=MDD(my_rendimenti,index,complete=False)
    volatility=my_rendimenti.std()
    final_cum=my_rendimenti.cumsum()[-1]
    
    equi=np.repeat(1/num_title,num_title).reshape(-1,num_title)
    equi=np.repeat(equi,len(rendimenti),axis=0)
    equi_rendimenti=np.sum(equi*rendimenti,axis=1)
    equiSR=sharpe_ratio(equi_rendimenti, rf)
    equisortino=sortino(equi_rendimenti, index,rf,complete=False)
    equiMDD=MDD(equi_rendimenti,index,complete=False)
    equivolatility=equi_rendimenti.std()
    finalequi=equi_rendimenti.cumsum()[-1]
    
    print("Final result",final_cum, "\n",
          "Final result portfolio equidistribuito",finalequi, "\n",
          "Sharpe ratio RL portfolio",mySR, "\n",
          "Sharpe ratio portafolio quidistribuito", equiSR,"\n",
          "Sortino RL portfolio",mysortino,"\n",
          "Sortino portfolio quidistribuito",equisortino,"\n",
          "MDD RL portfolio", my_mdd,"\n",
          "MDD equi portfolio", equiMDD,"\n",
          "Volatility RL portfolio",volatility, "\n",
          "Volatility equi portfolio",equivolatility
          )
    return(array([final_cum,finalequi,mySR,equiSR,mysortino,equisortino,my_mdd,equiMDD,volatility,equivolatility]))

 
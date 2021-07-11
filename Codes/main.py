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
from functions.generic_functions import *
from functions.financial_functions import *
from functions.rl_functions import *

#############################
## Hyperparameters
#############################
lay = list((3,256,128,64));          # Neural Network dimension: 3 layers, (256,128,64) neurons 
actions = (-1,0,1)                   # Possible actions for the financial problem (Short, Null, Long position)
back = 3                             # Number of return states t: R(t)={r(t),r(t-1),r(t-2)}
gamma = 0.8                          # Discount rate of the RL algorithm
espilon = 0.09                       # Exploration rate
type_exploration = "costant"         # Type of exploration rate decay (exp, quadratic, cubic, costant)
loss_function = "huber_loss"         # Loss function
batch_size = 256                     # Mini-batch dimension
num_episodes = 500                   # Number of episodes
update_target_net = 30000            # Step needed for an hard update of the target network 
reset_memory = 90000                 # Total capacity of experience memory
condition_update = 90                # Update of the target network after 90 new observations


commission=0.0025                    # Commission to change position on the stock, computed as percentage of the price of the stock = 0.25%
test_years=1                         # Size in year of the test set
window=14                            # Temporal window for computing the RSI index
entity=1                             # Parameter needed for the portfolio construction

path_save="~/model/"


################################
# Download data
################################
name=("goog","t","ibm","wba","vz","axp","trv","xom","dis","f","nsrgy")
prezzi_close_train,returns_train,others_train=download(name, window,test_years)

###############################
# Model estimation
###############################
index=len(returns_train[0])-test_years*252 #train and test split
y=DeepQLearning(returns_train.copy(),lay=lay,actions=actions, length_obs=252*test_years, transaction_cost=commission, batch_size=batch_size,
                other=others_train.copy(), index=index, back_states=back,episodes=num_episodes, loss_function=loss_function, decay_mod=type_exploration,
                learning_rate=0.0001,name=name,epsilon_init=espilon, gamma_init=gamma, reset_memory=reset_memory, 
                update_target_net=update_target_net,condition_update=condition_update)


##############################
# Validation and convergence
##############################
name_val=("tsla","msft","rl")
X=validation(name_val,path_save,5,num_episodes,commission=commission, test_years=test_years, back=back, window=window)
X.plot()
X.to_csv(path_save+"validation.csv")
num=str(np.argmax(X.mean(axis=1))*5)
print(num)

X=pd.read_csv(path_save+"validation.csv",index_col=0)
graph_analysis(X,5)

#############################
# Results on train set
#############################
final_res=list();misure=list()
for k in range(len(name)):
  model=keras.models.load_model(path_save+num)
  plt.figure(k,figsize=(8, 5))
  final_res.append(model_results(returns_train[k],model,back,actions,raw_price=prezzi_close_train[k],name=name[k], 
                                 commission=commission,test_index=index,other_data=others_train[k]))
  plt.show()
  plt.figure(figsize=(5, 3))
  misure.append(misure_val(final_res[k], index, name[k], commission=commission))
  plt.show()

tests(misure)

#############################
# Results on test
#############################
name_test=("ba","dal","ag","ing","mt","nls","per","db","wfc",
           "nke","wmt","hd","mcd","pfe","csco","aapl","ko","unh","tsla","msft",
           "gs","sne","jpm","mmm","aon","adbe","fdx","swk","rl","ma","v","disca",
           "nvda")
prezzi,returns,others=download(name_test)

final_res_test=list();misure_test=list()
for k in range(len(name_test)):
  model=keras.models.load_model(path_save+num)
  plt.figure(k,figsize=(7, 5))
  final_res_test.append(model_results(returns[k],model,back,actions,name=name_test[k], raw_price=prezzi[k],
                                      commission=commission, test_index=index,other_data=others[k]))
  plt.show()
  plt.figure(figsize=(5, 3))
  misure_test.append(misure_val(final_res_test[k], index, name_test[k], commission=commission))
  plt.show()

tests(misure_test)

k=pd.DataFrame(round(misure_test[0],3))
for i in range(1,len(misure_test)):
  k=pd.concat([k,pd.DataFrame(round(misure_test[i],3))],axis=1)
  k.to_csv(path_save+"misure_test.csv")

############################
# Building portfolio
############################
name="unh"
start_date=pd.to_datetime("2012-01-25")
end_date=pd.to_datetime("2020-01-01")
data=yf.download(name, start=start_date, end=end_date)
dates=data.index

start_date=pd.to_datetime("2019-01-01")
data=yf.download(name, start=start_date, end=end_date)
dates_test=data.index

real_name=("Google","Ibm","AT&T","Wba","Verizon","Axp","Trv","Exxon","Ford","Disney","Nestlé")
portfolio_weights,My_port,Equi_port,Random_port, Rendimenti,rlret,comm=build_portfolio(final_res,real_name,dates,commission,entity)
valuta_port(portfolio_weights,Rendimenti,rlret)

# More importance to the actions
portfolio_weights,My_port,Equi_port,Random_port, Rendimenti,rlret,comm=build_portfolio(final_res,real_name,dates,commission,entity=5)
valuta_port(portfolio_weights,Rendimenti,rlret)


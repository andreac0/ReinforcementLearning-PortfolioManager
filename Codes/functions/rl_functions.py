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
# Function to compute action based on the state
###################################
def argmaxQnet(model,actions,num_input,st):
    q_value = model.predict(st.reshape(-1,num_input))
    return(actions[np.argmax(q_value)])    

###################################
# Decay of exploration
###################################
def decay(epsilon, i, what,degree=5e-06):
    if what=="exp": return(0.06 + (epsilon-0.06)*np.exp(-degree*(i+1)))
    if what=="costant": return(epsilon)
    if what=="quadratic":return(1/(1+i)**(1/2))
    if what=="cubic": return(1/(1+i)**(1/3))


###################################
# Build the neural network
###################################
def actionvalueNetwork(layers,num_input,loss_function, learning_rate, num_actions=3):
    model = Sequential()
    model.add(Dense(layers[1], activation='relu',input_dim=num_input))
    for k in range(2,layers[0]+1):
     model.add(Dense(layers[k], activation='relu'))
    model.add(Dense(units=num_actions, activation='linear'))
    opt=tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss=loss_function, optimizer=opt)
    return(model)


########################################
# FUNZIONE DI STIMA DEL MODELLO
########################################
def DeepQLearning(returns, actions, back_states=3,other=np.array([]), gamma_init=0.8, episodes=250,length_obs=252, 
                  loss_function='mse', index=1000, learning_rate=0.0001,lay=list((3,120,60,30)), decay_mod="costant",
                  name="Data", epsilon_init=0.1, type_reward="simple", transaction_cost=0.0025, batch_size=256,
                  update_target_net=9000,reset_memory=90000, condition_update=60):

    print('data','=',name,'\n',
          'episodes','=',episodes,'\n',
          'back state','=',back_states,'\n',
          'gamma','=',gamma_init, '\n',
          'exploration epsilon decay','=',decay_mod, '\n',
          'batch size','=',batch_size,'\n',
          'learning rate','=',learning_rate,'\n',
          'obs. per episode','=',length_obs, '\n',
          'layers','=',lay,'\n',
          'loss function','=',loss_function,'\n',
          'reward function','=', type_reward, '\n',
          'transaction cost','=',transaction_cost,'\n',
          'step to update the target netowrk','=',update_target_net,'\n',
          'step to reset memory','=',reset_memory,'\n',
          'step to update','=',condition_update,'\n',
          'index test', '=', index)
    
    #Standardizzo i rendimenti e la commission in modo da renderla confrontabile
    altre_var=list()
    commission_cost=np.zeros(len(returns))
    for i in range(len(returns)):
      returns[i],commission_cost[i]=standardize(returns[i], index,commission=transaction_cost, ret=True)
      returns[i]=returns[i][:index]
      altre_var.append(other[i][:index])

    #Inizializzazione alcune variabili utili
    dataset=list()
    gamma=gamma_init
    num_obs,num_input=len(returns[0]),(back_states+other[0].shape[1]+1)
 
        #Costruisco la rete e la rete target
    model=actionvalueNetwork(lay,num_input,loss_function, learning_rate, num_actions=len(actions))
    model_target=tf.keras.models.clone_model(model); model_target.set_weights(model.get_weights())
    
    #CICLO DEGLI EPISODI
    for h in range(0,episodes+1):
     filepath='/content/'+path_save+str(h)  
     if h%5==0: model.save(filepath)                                    #Il modello viene salvato ogni 5 episodi, in modo da avere dei backup per valutare successivamente la convergenza
        
        #Tasso di esplorazione
     epsilon=decay(epsilon_init, h, decay_mod)

        #Scelta dell'asset dell'episodio
     K=np.random.choice(range(len(returns)))
     ret=returns[K]
     other_data=altre_var[K]

        #Scelta del giorno di partenza dell'episodio
     x=array(range(back_states,num_obs-length_obs))
     x=softmax(x/len(x))
     init_obs=int(np.random.choice(range(back_states,num_obs-length_obs),size=1,p=x))
     
      #Inizializzo vettore delle azioni per motivi computazionali
     azione=np.random.choice(actions,back_states+1, replace=True)
     
     #Ciclo sulla finestra temporale
     for i in range(init_obs+1,init_obs+length_obs):
         #Compute S_{t-1} 
       st=np.append(ret[(i-back_states):i],np.append(other_data[i-1,:],azione[-1])).reshape(-1,num_input)
         #Take action with epsilon-greedy policy
       azione=np.append(azione,np.random.choice(np.append(argmaxQnet(model,actions,num_input,st),actions),1,p=np.append(1-epsilon,np.repeat(epsilon/len(actions),len(actions)))))
         #Compute S_{t}
       st1=np.append(ret[(i-back_states+1):(i+1)],np.append(other_data[i,:],azione[-1])).reshape(-1,num_input)
         
         #Simula le altre azioni
       if azione[-1]==1: azione_target,azione_target2=-1,0
       elif azione[-1]==0:  azione_target,azione_target2=1,-1
       else:  azione_target,azione_target2=0,1

         #Target States
       st_t=np.append(ret[(i-back_states):i],np.append(other_data[i-1,:],azione[-2])).reshape(-1,num_input)
       st1_t=np.append(ret[(i-back_states+1):(i+1)],np.append(other_data[i,:],azione_target)).reshape(-1,num_input)

       st_t2=np.append(ret[(i-back_states):i],np.append(other_data[i-1,:],azione[-2])).reshape(-1,num_input)
       st1_t2=np.append(ret[(i-back_states+1):(i+1)],np.append(other_data[i,:],azione_target2)).reshape(-1,num_input)

         #Compute reward
       reward=azione[-1]*ret[i]-(azione[-1]!=azione[-2])*commission_cost[K]
       reward_target=(azione_target*ret[i])-(azione_target!=azione[-2])*commission_cost[K]
       reward_target2=(azione_target2*ret[i])-(azione_target2!=azione[-2])*commission_cost[K]

        #Add datas to memory
       dataset.append((st,azione[-1],reward,st1))
       dataset.append((st_t,array(azione_target),reward_target,st1_t))
       dataset.append((st_t2,array(azione_target2),reward_target2,st1_t2))
       
        #Update the model 
       L=len(dataset)
       if (L%condition_update==0 and L>batch_size):
        sampleD=np.random.choice(range(0,L),batch_size,replace=False)
        stato1,stato2,r,a=array([]).reshape(-1,num_input),array([]).reshape(-1,num_input),array([]),array([])
        for t in sampleD:
          stato1,stato2=np.append(stato1,dataset[t][0].reshape(-1,num_input),axis=0),np.append(stato2,dataset[t][3].reshape(-1,num_input),axis=0)
          a,r=np.append(a,int(array(np.where(actions==dataset[t][1])))),np.append(r,dataset[t][2])
        
        mod,mod_target=model.predict(stato2),model_target.predict(stato2)
        positions=np.argmax(mod,axis=1)   

        for t in range(len(positions)):
          mod[t][int(a[t])]=r[t]+gamma*mod_target[t][positions[t]]
        
        model.fit(stato1,mod, epochs=1, verbose=0)

        if L%update_target_net==0: model_target.set_weights(model.get_weights())
        if L%reset_memory==0: dataset=list();  

     print('episode:',h+1)
    return(model)

   ##############################
   # Convergence evaluation and validation
   ##############################
def val_results(ret, model,back,actions,test_index,name,raw_price,other_data=array([]),commission=0.0025):
    ret2=copy.deepcopy(ret)
    ret=standardize(ret, test_index)
    azione=np.repeat(array([0]),back) 
    num_input=(back+other_data.shape[1]+1)
    weights=np.repeat(0,back)
    for i in range(back,test_index):
        q_value=model.predict(np.append(ret[i-back:i],np.append(other_data[i-1,:],azione[-1])).reshape(-1,num_input))
        azione=np.append(azione,actions[np.argmax(q_value)])
        weights=np.append(weights,np.amax(q_value))
    returns=array([])
    for i in range(len(azione)):
        returns=np.append(returns,azione[i]*ret2[i]-commission*(azione[i]!=azione[i-1]))
    cumreturns=returns.cumsum()
    buy_hold=ret2[:test_index].cumsum()
    
    complete_series=np.concatenate((raw_price[1:test_index+1].reshape(-1,1),ret2[:test_index].reshape(-1,1),azione.reshape(-1,1),cumreturns.reshape(-1,1),buy_hold.reshape(-1,1)),axis=1)
    return(complete_series)

def validation(name,percorsofile,intervallo,modfin,data_inizio="2012-01-01",data_fin="2020-01-01",commission=0.0025, test_years=1, back=3, window=14):
 actions=(-1,0,1)
 validation=list(); lista_sharpe=list()
 price,returns,others=download(name)
 for i in range(len(name)):
  sr=array([])
  for k in range(0,modfin,intervallo):
    model=keras.models.load_model(percorsofile+str(k))
    validation.append(val_results(returns[i],model,back,actions,name=name,raw_price=price[i], commission=commission,
                      test_index=index,other_data=others[i]))
    sr=np.append(sr,sharpe_ratio(validation[-1][:,1]*validation[-1][:,2],rf=0.01/100))
  lista_sharpe.append(sr)
  print(i)

 d=np.concatenate((lista_sharpe[0].reshape(-1,1),lista_sharpe[1].reshape(-1,1),lista_sharpe[2].reshape(-1,1)),axis=1)
 d=pd.DataFrame(d)
 d.mean(axis=1).plot()
 return(d)

def graph_analysis(X, interval_episodes):
  yy=X.mean(axis=1)
  xx=array(range(len(X))).reshape(-1,1)
  poly = PolynomialFeatures(degree = 2) 
  X_poly = poly.fit_transform(xx) 
  
  poly.fit(X_poly, yy) 
  lin2 = LinearRegression() 
  lin2.fit(X_poly, yy) 

  plt.scatter(xx*interval_episodes,yy, color = 'blue',s=10) 
  plt.plot(xx*interval_episodes, lin2.predict(poly.fit_transform(xx)), color = 'green', linewidth=3) 
  plt.show() 

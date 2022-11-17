import numpy as np
import pandas as pd
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler as lr_scheduler
#import torch.optim.lr_scheduler.ReduceLROnPlateau 
from torch.nn import functional as F
from torchvision import datasets, transforms
import function_files as ff

from scipy.stats.kde import gaussian_kde
from numpy import linspace
import matplotlib.pyplot as plt

#%%
#importing the data 
data_1_load= pd.read_csv("y_data1.csv")
data_2_load= pd.read_csv("y_data2.csv")
label_d = pd.read_csv("label_data.csv")
t_d= pd.read_csv("t_data.csv")

N = 2000  # number of time series
tp = 100   # time steps 

# train and validation data
data_1=torch.tensor(np.array(data_1_load)[0:(N-200)*tp,:])
data_2=torch.tensor(np.array(data_2_load)[0:(N-200)*tp,:])
label_d1=torch.tensor(np.array(label_d)[0:(N-200)*tp,:])
data_1_t=torch.tensor(np.array(data_1_load)[(N-200)*tp:N*tp,:])
data_2_t=torch.tensor(np.array(data_2_load)[(N-200)*tp:N*tp,:])
label_dt=torch.tensor(np.array(label_d)[(N-200)*tp:N*tp,:])

# data loader
data_1=DataLoader(data_1, batch_size=100, shuffle=False)
data_2=DataLoader(data_2, batch_size=100, shuffle=False)
label_d=DataLoader(label_d1, batch_size=100, shuffle=False)


#importing Testing data
import h5py
f = h5py.File('data1u.mat','r')
mat01 = f.get('mat011u')
mat01 = np.array(mat01)
mat02 = f.get('mat012u')
mat02 = np.array(mat02)
mat03 = f.get('mat013u')
mat03 = np.array(mat03)

test_tr1=torch.t(torch.tensor(mat01))
test_tr2=torch.t(torch.tensor(mat02))
label_tr=torch.t(torch.tensor(mat03))

#%%Model
num_epochs = 250
batch_size = 100
learning_rate = 1e-5
num_features =2
num_latents =32
num_labels =4

model = ff.model_autoencoder(num_features, num_latents, num_labels)
ff.train(model, data_1, data_2, label_dt, num_epochs= 500, shedule=True, criterion =nn.MSELoss(), title=None)

#%% Prediction
predict2=[];
for i in range(0,10000):
   predict1= ff.prediction_data(model, torch.tensor(np.array(test_tr1)[i*100,:]),torch.tensor(np.array(label_tr)[i*100:i*100+100,:]))
   predict1=np.array(predict1)
   predict2.append(predict1)
#%%computing reliability (failure probabilty)
data_tr1= np.array(test_tr1).reshape(10000,100,2)
data_tr2= np.array(test_tr2).reshape(10000,100,2)
label_tr= np.array(label_tr).reshape(10000,100,4)

#test
p_f = ff.failurefunction(data_tr2[:,:,0],2.5,10000,40)
print(p_f)

# prediction
predict2=np.array(predict2)
p_f_p = ff.failurefunction(predict2[:,:,0],2.5,10000,40)

#%% PDF of the fiorst passagea failure time
k1= ff.densityfunction(predict2[:,:,0],2.5,10000,100)
k2= ff.densityfunction(data_tr2[:,:,0],2.5,10000,100)

t_data=np.array(t_d)[1:101]

kde = gaussian_kde(t_data[k1,0] )
dist_space = linspace( -2, 6, 100 )
plt.plot( dist_space, kde(dist_space) )
kde = gaussian_kde(t_data[k2,0] )
dist_space = linspace(-2, 6, 100 )
plt.plot( dist_space, kde(dist_space) )

np.savetxt('tf1.txt',t_data[k1 ,0])
np.savetxt('tf2.txt',t_data[k2,0])

#importing packages
import numpy as np
import pandas as pd
#from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler as lr_scheduler
#import torch.optim.lr_scheduler.ReduceLROnPlateau 
from torch.nn import functional as F
from torchvision import datasets, transforms
import  math
# In[]:function to define the model
def model_autoencoder(latent):
  class autoencoder(nn.Module):
    def __init__(self,latent):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 500),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(500,latent))
            
        
        self.koopman=nn.Sequential(
            nn.Linear(latent,latent,bias=False))
        
        self.decoder = nn.Sequential(
            nn.Linear(latent, 500),
            nn.Dropout(p=0.1),
            nn.ReLU(),  
            nn.Linear(500, 2))


    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.koopman(x1)
        x3 = self.decoder(x2)
        return x1, x2, x3

  model_d=autoencoder(latent)  
  return model_d
#%% Train the koopman model
def train(model, dataloader1, dataloader2, num_epochs=10, shedule=True, criterion =nn.MSELoss(), title=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
    train_losses = []
    if shedule==True:
      scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    for epoch in range(num_epochs):
        for (data1,data2) in zip(dataloader1,dataloader2):
              output1, output2, output3 = model(data1.float())
              loss = criterion( data2.float(), model.decoder(model.koopman(model.encoder(data1.float()))))+ \
                  criterion(output2,model.koopman(output1))+ criterion(data1.float(), model.decoder(model.encoder(data1.float())))
              optimizer.zero_grad() 
              loss.backward()
              optimizer.step()
              #scheduler.step()
              #train_losses.append(loss.item())

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))    
#%%Prediction of koopman 
def prediction_data1(model,x0):
   model.eval()
   predict=[];
   with torch.no_grad():
      data_t1=x0
      for i in range(1,101):
       t1, t2, data_t2=model(data_t1.float())
       data_t1=data_t2
       predict.append(data_t2.numpy())
   return predict
# In[]:
def get_embeddings(model,x0):
   model.eval()
   embedding=[];
   with torch.no_grad():
      for i in range(0,x0.shape[0]):
       embedd1 = model.encoder(x0[i,:].float())
       embedding.append(embedd1.numpy())
   return torch.tensor(embedding)
# In[]:
def failurefunction(y0,Lt,N,T_step):
    count=0;
    y1=y0[:,0:T_step];
    for i in range(0,N):
       ja1=np.where(abs(np.array(y1)[i])>Lt)[0]
       ja=ja1.tolist()
       if len(ja)!=0 :      
          count=count+1
    Pf=count/N
    return Pf
#%%
def densityfunction(y0,Lt,N,T_step):
    index_f=(T_step-1)*np.ones(N,dtype=int);
    y1=y0[:,0:T_step];
    for i in range(0,N):
       ja1=np.where(abs(np.array(y1)[i])>Lt)[0]
       ja=ja1.tolist()
       if len(ja)!=0 :      
         index_f[i]=min(ja)
    return index_f


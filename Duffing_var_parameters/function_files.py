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
def model_autoencoder(num_features, num_latents, num_labels):
  class autoencoder(nn.Module):
    def __init__(self,num_features, num_latents, num_labels):
        super(autoencoder, self).__init__()
        self.num_labels=num_labels
        self.num_feature=num_features
        self.num_latent=num_latents

        self.fc11=nn.Linear(num_features+num_labels, 500)
        self.fc12=nn.Linear(500,num_latents)

        self.fc2=nn.Linear(num_latents, num_latents, bias=False)

        self.fc31=nn.Linear(num_latents+num_labels, 500)
        self.fc32=nn.Linear(500,num_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        #self.laynorm= nn.LayerNorm(32, eps=1e-05)
       
       
    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, num_features)
        c: (bs, num_labels)
        '''
        if (x.ndim)==1:
          inputs = torch.cat([x, c], 0) # (bs, feature_size+class_size)
        else :
          inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        
        h1 = self.relu(self.fc11(inputs))
        h1=self.dropout(h1)
        x1 = self.fc12(h1)
        return x1

    def koopman(self, x1): # P(x|z, c)
        '''
        x1: (bs, num_latents)
        c: (bs, num_labels)
        '''
        inputs = x1 # (bs, latent_size+class_size)
        x2 = self.fc2(inputs)
        return x2

    def decode(self, x2, c): # P(x|z, c)
        '''
        x2: (bs, num_latents)
        c: (bs, num_features)
        '''
        if (x2.ndim)==1:
          inputs = torch.cat([x2, c], 0) # (bs, feature_size+class_size)
        else : 
          inputs = torch.cat([x2, c], 1) # (bs, feature_size+class_size)
        h3 = self.relu(self.fc31(inputs))
        h3=self.dropout(h3)
        x3= self.fc32(h3)
        return x3

       

    def forward(self, x,c):
        x1 = self.encode(x, c)
        x2 = self.koopman(x1)
        x3 = self.decode(x2,c)
        return x1, x2, x3
#%% Train the koopman model
def train(model, data_1, data_2, label_d, num_epochs, shedule=True, criterion =nn.MSELoss(), title=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
    train_losses = []
    if shedule==True:
      scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    for epoch in range(num_epochs):
        for (data1,data2, c11) in zip(data_1,data_2,label_d):
            output1, output2, output3 = model(data1.float(), c11.float())
            loss = criterion(data2.float(), model.decode(model.koopman(model.encode(data1.float(),c11.float())),c11.float()))+ criterion(model.decoder(data2),model.koopman(output1.float()))+ criterion(data1.float(), model.decode(model.encode(data1.float(),c11.float()),c11.float()))
            # ===================forward=====================
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))    
#%%Prediction of koopman 
def prediction_data1(model,x0,c0):
   model.eval()
   predict=[];
   with torch.no_grad():
      data_t1=x0
      label_t1=c0
      for i in range(0,100):
        c_l=torch.tensor(np.array(label_t1)[i,:]) 
        t1, t2, data_t2=model(data_t1.float(), c_l.float())
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








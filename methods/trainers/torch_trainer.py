# Local Imports
from trainers.base_trainer import Trainer
from util.data_loader import prep_dataset

# Package Imports
from abc import ABC
import gc
import torch
import numpy as np
from sklearn.metrics import r2_score
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class TorchTrainer(Trainer,ABC):
  
  stop_patience = 15
  lr_patience = 10

  device = "cuda" if torch.cuda.is_available() else "cpu"
 
  def load_data(self,split):
    super(TorchTrainer,self).load_data(split) 

    self.X_train = torch.tensor(self.X_train,device = self.device,dtype=torch.long)
    self.y_train = torch.tensor(self.y_train,device = self.device,dtype=torch.float32)
    self.X_val = torch.tensor(self.X_val,device = self.device,dtype=torch.long)
    self.y_val = torch.tensor(self.y_val,device = self.device,dtype=torch.float32)
    self.X_test = torch.tensor(self.X_test,device = self.device,dtype=torch.long)
    self.y_test = torch.tensor(self.y_test,device = self.device,dtype=torch.float32)
    
     

  def fit_model(self,model):
   

    curr_time = time.time() 
    training_data = torch.utils.data.TensorDataset(self.X_train,self.y_train)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=2**self.bs,shuffle = True)
    
    val_data = torch.utils.data.TensorDataset(self.X_val,self.y_val)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=2**self.bs)
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.LR,weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor = 0.5,patience = self.lr_patience,threshold = 1e-3)
    early_stopping = EarlyStopper(patience=self.stop_patience,
                                                   min_delta = 0.001)
    loss_fn = torch.nn.MSELoss()

    size = len(train_dataloader.dataset)
   
    model.train()


    y_pred = []
    y_true = []
    min_loss = np.inf
    for epoch in range(self.EPOCHS):
      
      print()
      y_pred = []
      y_true = []
      batch_loss = 0
      batch_r2 = 0
      for batch, (X,y) in enumerate(train_dataloader):
        X,y = X.to(self.device), y.to(self.device)
          
        pred = model(X)
        y_true.append(y)
        y_pred.append(pred)
        loss = loss_fn(pred,y)
        batch_loss += loss
        batch_r2 += r2_score(y.cpu().detach().numpy(),pred.cpu().detach().numpy())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if self.VERBOSE == 1:
          print("Epoch {} | {}% | Loss {:.4f} | R2 {:.4f}".format(epoch,int(100*batch/size*(2**self.bs)),batch_loss/(batch+1),batch_r2/(batch+1)),end = '\r')
 
      y_pred = torch.cat(y_pred)
      y_true = torch.cat(y_true)
      train_loss = loss_fn(y_pred,y_true)
      train_r2 = r2_score(y_true.cpu().detach().numpy(),y_pred.cpu().detach().numpy())

      model.eval()
      y_pred = []
      y_true = []
      
      with torch.no_grad():
        for X,y in val_dataloader:
          X,y = X.to(self.device),y.to(self.device)
          y_pred.append(model(X))
          y_true.append(y)
      y_pred = torch.cat(y_pred)
      y_true = torch.cat(y_true)
      val_loss = loss_fn(y_pred,y_true)
      val_r2 = r2_score(y_true.cpu().detach().numpy(),y_pred.cpu().detach().numpy())
      scheduler.step(train_loss)
 

      if self.VERBOSE == 1: 
          print("Epoch {} | 100% | Loss {:.4f} | R2 {:.4f} | Val Loss {:.4f} | Val R2 {:.4f}".format(epoch,train_loss,train_r2,val_loss,val_r2),end = '\r')
 
      if val_loss < min_loss:
        min_loss = val_loss
        torch.save(model.state_dict(),"./torch_max_weights_{}.pkl".format(curr_time))   
 
      if early_stopping.early_stop(val_loss):
        model.load_state_dict(torch.load("./torch_max_weights_{}.pkl".format(curr_time)))
        break
      

  def pred_model(self,model,test):
    if test:
      X = self.X_test
      y = self.y_test
    else:
      X = self.X_train
      y = self.y_train
    
    data = torch.utils.data.TensorDataset(X,y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=2**self.bs)

    model.eval()

    y_pred = []
    y_true = []
    with torch.no_grad():
      for X,y in dataloader:
        X,y = X.to(self.device),y.to(self.device)
        y_pred.append(model(X))
        y_true.append(y)
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    y_pred = y_pred*self.std + self.mean
    y_true = y_true*self.std + self.mean
    
    return y_true.cpu().numpy(),y_pred.cpu().numpy()
 
  def train(self,split,prune = True,return_model = False):


    self.load_data(split)
    self.model = self.create_model()    
    self.fit_model(self.model)

    self.model.train()
    #X_train = TorchTrainer.unstack_x(self.X_train.cpu().numpy())
    #y = TorchTrainer.unstack_y(self.y_train.cpu().numpy())

    #pair_embed = self.model.get_pair_embed().detach().cpu().numpy()
    #pair_embed = pair_embed[:,:,np.std(pair_embed,axis = (0,1)) != 0]
    #np.save('pair_embed.npy',pair_embed)
    #pair_embed = np.concatenate([pair_embed[X_train[:,0],X_train[:,2],:],pair_embed[X_train[:,1],X_train[:,2],:]],axis = -1)
    #pair_embed = TSNE(2).fit_transform(PCA(50).fit_transform(pair_embed))
    #pair_embed = PCA(2).fit_transform(pair_embed)
    
    #color = np.zeros((pair_embed.shape[0],3))
    
    #y -= y.mean()
    #y /= y.std()
    #y_cpy = y.copy()
    #y_cpy[y>1] = 1
    #y_cpy[y<-1] = 0
    #y_cpy[np.logical_and(y>=-1,y<=1)] = 0.5
    #y = y_cpy
  
    #color[:,0] = y

 
    #plt.scatter(pair_embed[:,0],pair_embed[:,1],s=1,c = y,marker = '*',norm = None,alpha = 0.5,cmap='plasma')
    #plt.savefig('./pca_plot.png',dpi = 1024)
#
    y_true_split,y_pred_split = self.pred_model(self.model,test=True)
    y_true_train,y_pred_train = self.pred_model(self.model,test=False)
  
    if prune:
      self.prune(split,y_true_split,y_pred_split)  

    return y_true_split,y_pred_split,y_true_train,y_pred_train
  
  def create_model(self):
    gc.collect()
    torch.cuda.empty_cache()
    model = self.model_class(**self.param,**self.hparam)
    model.to(self.device)
    return model
  
  def save_weights(self,path):
    torch.save(self.model.state_dict(),path+".pkl")   
  

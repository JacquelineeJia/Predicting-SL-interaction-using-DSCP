# Local Imports
from models import HyperGraphSynergy
from trainers.torch_trainer import TorchTrainer,EarlyStopper
from util.data_loader import read_aux,read_sim
from rdkit import Chem
import pandas as pd
import numpy as np
import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, mean_squared_error, r2_score
from scipy.stats import pearsonr
import random
import deepchem as dc
from torch_geometric.data import Data,Batch
from sklearn.metrics import r2_score
import time
import gc


def getData(dataset):
   
  try:
    drug_graph_data = np.load('/home/judah/dscp/data/{}/drug_graph.npy'.format(dataset),allow_pickle = True)
  except:
    drug_smiles_file = '/home/judah/dscp/data/{}/smi.npy'.format(dataset)
    drug = np.load(drug_smiles_file)
    featurizer = dc.feat.MolGraphConvFeaturizer()
    drug_graph_data = featurizer.featurize([sm for sm in drug])
    np.save('/home/judah/dscp/data/{}/drug_graph.npy'.format(dataset),drug_graph_data,allow_pickle = True)

  pyg_graph = []
  for graph in drug_graph_data:
    if not isinstance(graph,np.ndarray):
      pyg_graph.append(Data(graph.node_features,graph.edge_index))
   

  return  Batch.from_data_list(pyg_graph)


class HyperGraphSynergyTrainer(TorchTrainer):

  model_class = HyperGraphSynergy
  nn_choices = [[8192,8192],[4096, 4096], [2048, 2048], [8192, 4096], [4096, 2048], [4096, 4096, 4096], [2048, 2048, 2048], [4096, 2048, 1024], [8192, 4096, 2048]] 

  def __init__(self,dataset):
    super(HyperGraphSynergyTrainer,self).__init__(dataset)

    _,self.param['cl_feat'] = read_aux(dataset,pca = False,norm = True)
    self.param['drug_sim'],self.param['cl_sim'] = read_sim(dataset)
    self.param['shape'] = self.shape 
    self.param['drug_graph'] = getData(dataset)

  def set_trial(self,trial):
    super(HyperGraphSynergyTrainer,self).set_trial(trial)

    self.alpha = trial.suggest_float('alpha',0,0,step = 0.1) 
    self.hparam['nn'] = trial.suggest_categorical('nn',self.nn_choices)
    self.hparam['hidden_do'] = trial.suggest_float('hidden_do',0.3,0.3,step = 0.1)
    self.hparam['out_do'] = trial.suggest_float('out_do',0.1,0.1,step = 0.1) 
 
   
  def load_data(self,split):
    super(HyperGraphSynergyTrainer,self).load_data(split)

    self.param['X_train'] = self.X_train.detach().cpu().numpy()
    self.param['y_train'] = self.y_train.detach().cpu().numpy()

 
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
          
        pred,sim_loss = model(X)
        y_true.append(y)
        y_pred.append(pred)
        loss = loss_fn(pred,y)*(1-self.alpha) + sim_loss*self.alpha
        batch_loss += loss_fn(pred,y)
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
          y_pred.append(model(X)[0])
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
        y_pred.append(model(X)[0])
        y_true.append(y)
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    y_pred = y_pred*self.std + self.mean
    y_true = y_true*self.std + self.mean
    
    return y_true.cpu().numpy(),y_pred.cpu().numpy()
  
  def set_hparam(self,path):
    self.hparam = np.load(path,allow_pickle = True).item()
    self.bs = self.hparam['bs']
    self.alpha = self.hparam['alpha']
    del self.hparam['bs'] 
    del self.hparam['alpha'] 
  

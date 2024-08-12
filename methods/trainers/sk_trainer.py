# Local Imports
from trainers.base_trainer import Trainer
from util.met import tf_r2
from util.data_loader import prep_dataset,read_aux


# Package Imports
from abc import ABC
import gc
import numpy as np

class SKTrainer(Trainer,ABC):
    

  def __init__(self,dataset):
    super(SKTrainer,self).__init__(dataset)
    
    self.param['verbose'] = 2
    self.param['drug_feat'],self.param['cl_feat'] = read_aux(dataset,
                                                    pca = False,
                                                    norm = True,
                                                    num_cl_pc = self.shape[-1],
                                                    num_drug_pc = self.shape[0])
  

  def create_model(self):
    gc.collect()
    param = self.param.copy()
    del param['drug_feat']
    del param['cl_feat']
    return self.model_class(**param,**self.hparam)

  def load_data(self,split):
    super(SKTrainer,self).load_data(split)

    self.X_train = np.concatenate([self.param['drug_feat'][self.X_train[...,0]],
                                   self.param['drug_feat'][self.X_train[...,1]],
                                     self.param['cl_feat'][self.X_train[...,2]]],axis = -1)
    self.X_val = np.concatenate([self.param['drug_feat'][self.X_val[...,0]],
                                 self.param['drug_feat'][self.X_val[...,1]],
                                   self.param['cl_feat'][self.X_val[...,2]]],axis = -1)
    self.X_test = np.concatenate([self.param['drug_feat'][self.X_test[...,0]],
                                  self.param['drug_feat'][self.X_test[...,1]],
                                    self.param['cl_feat'][self.X_test[...,2]]],axis = -1)
   
 
  def fit_model(self,model):
      
    model.fit(self.X_train,self.y_train)

  def pred_model(self,model,test = True):
    if test:
      X = self.X_test
      y = self.y_test
    else:
      X = self.X_train
      y = self.y_train
    y_test_pred = model.predict(X)
    y_pred = y_test_pred*self.std + self.mean
    y_true = y*self.std + self.mean

    return y_true,y_pred
 
  def train(self,split,prune = True,return_model = False):


    self.load_data(split)
    model = self.create_model()    
    self.fit_model(model)
    y_true_split,y_pred_split = self.pred_model(model)
    y_true_train,y_pred_train = self.pred_model(model,test=False)
    
    if prune:
      self.prune(split,y_true_split,y_pred_split)  

    if return_model:
      class model():
        def save_weights(self,path):
          pass
      return y_true_split,y_pred_split,y_true_train,y_pred_train,model()
    else:
      return y_true_split,y_pred_split,y_true_train,y_pred_train
 
  def save_weights(self,path):
    pass 

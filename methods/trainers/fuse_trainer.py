# Local Imports
from trainers.tf_trainer import TFTrainer
from models import Fuse
from util.data_loader import read_tensor
from util.data_loader import read_aux

# Package Imports
import numpy as np

class FuseTrainer(TFTrainer):
  model_class = Fuse
  nn_choices = [[1024, 1024, 512],
                [2048, 2048, 1024],
                [2048, 1024, 512],
                [2048, 2048, 2048],
                [1024, 1024, 1024],
                [2048, 2048],
                [1024, 1024],
                [512, 512],
                [1024, 512]]
  
  encoder_choices = [[2048, 1024, 512],
                     [1024,512,256],
                     [512,256,128],
                     [1024, 512],
                     [512, 256],
                     [256,128]]


  def __init__(self,dataset):
    super(FuseTrainer,self).__init__(dataset)

    self.X = read_tensor(dataset)
    self.param['drug_feat'],self.param['cl_feat'] = read_aux(dataset,pca = False,norm = True)
    self.param['shape'] = self.shape
  
  def set_trial(self,trial):
    super(FuseTrainer,self).set_trial(trial)

    self.hparam['rank'] = trial.suggest_int('rank',1,4) 
    self.hparam['aux_weight'] = trial.suggest_float('aux_weight',0,10) 
    self.hparam['low_dim'] = trial.suggest_int('low_dim',1,128)

    self.hparam['tensor_decoder_hidden_do'] = trial.suggest_float('tensor_decoder_hidden_do',0,0.5,step = 0.1)
    self.hparam['tensor_decoder_out_do'] = trial.suggest_float('tensor_decoder_out_do',0,0.2,step = 0.1)    
    self.hparam['tensor_decoder_nn'] = trial.suggest_categorical('tensor_decoder_nn',self.nn_choices) 
    
    self.hparam['encoder_nn'] = trial.suggest_categorical('encoder_nn',self.encoder_choices)
    self.hparam['encoder_out_do'] = trial.suggest_float('encoder_out_do',0,0.2,step = 0.1)   
    self.hparam['encoder_hidden_do'] = trial.suggest_float('encoder_hidden_do',0,0.5,step = 0.1)
    
    self.hparam['aux_decoder_hidden_do'] = trial.suggest_float('aux_decoder_hidden_do',0,0.5,step = 0.1)
    self.hparam['aux_decoder_out_do'] = trial.suggest_float('aux_decoder_out_do',0,0.2,step = 0.1)    


  def load_data(self,split,permute = False):
    super(FuseTrainer,self).load_data(split,permute)

    X_train_tensor = (self.X - self.mean)/self.std

    W = np.zeros_like(self.X)
    W[self.X_train[:,0],self.X_train[:,1],self.X_train[:,2]] = 1
    W[self.X_train[:,1],self.X_train[:,0],self.X_train[:,2]] = 1
    X_train_tensor = X_train_tensor*W

    self.param['X'] = X_train_tensor
  

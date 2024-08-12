# Local Imports
from trainers.rank_trainer import RankTrainer
from models import DSCP
from models import get_dnn
from models.attention import get_cross_attention_dnn
from util.data_loader import read_tensor

# Package Imports
import numpy as np

class DSCPTrainer(RankTrainer):
  model_class = DSCP
  nn_choices = [[1024, 1024, 512],
                [2048, 2048, 1024],
                [2048, 1024, 512],
                [2048, 2048, 2048],
                [1024, 1024, 1024],
                [2048, 2048],
                [1024, 1024],
                [512, 512],
                [1024, 512]]
  

  def __init__(self,dataset):
    super(DSCPTrainer,self).__init__(dataset)

    self.param['classifier'] = get_dnn

    #print
    print('We are now reading in the tensor.....')
    self.X = read_tensor(dataset)
    print('Success!')
    
  def set_trial(self,trial):
    super(DSCPTrainer,self).set_trial(trial)
    
    #self.hparam['activation_function'] = trial.suggest_categorical('activation_function', ['relu', 'elu', 'tanh'])
    self.hparam['activation_function'] = 'tanh' 
    self.hparam['l1_strength'] = trial.suggest_float('l1_strength', 1e-10, 1e-3, log=True)
    self.hparam['l2_strength'] = trial.suggest_float('l2_strength', 1e-10, 1e-3, log=True)

    self.hparam['hidden_do'] = trial.suggest_float('hidden_do',0,0.5,step = 0.1)
    self.hparam['out_do'] = trial.suggest_float('out_do',0,0.2,step = 0.1)
    
    self.hparam['nn'] = trial.suggest_categorical('nn',self.nn_choices) 
 
  def load_data(self,split,permute = False):
    super(DSCPTrainer,self).load_data(split,permute)

    X_train_tensor = (self.X - self.mean)/self.std
    print("=====X train tensor shape======")
    print(X_train_tensor.shape)

    W = np.zeros_like(self.X)
    print("W shape is: ", W.shape)
    print(self.X_train[:,0].shape)
    print("X trains shape: ", self.X_train.shape)
    W[self.X_train[:,0]-1,self.X_train[:,1]-1,self.X_train[:,2]-1] = 1
    W[self.X_train[:,1]-1,self.X_train[:,0]-1,self.X_train[:,2]-1] = 1
    X_train_tensor = X_train_tensor*W

    self.param['X'] = X_train_tensor
  

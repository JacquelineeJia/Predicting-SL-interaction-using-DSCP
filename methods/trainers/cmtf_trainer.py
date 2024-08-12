# Local Imports
from trainers.rank_trainer import RankTrainer
from util.data_loader import read_aux
from models import CMTF

# Package Imports
import numpy as np


class CMTFTrainer(RankTrainer):
  
  model_class = CMTF
  
  stop_patience = 50
  lr_patience = 30
 
  def __init__(self,dataset):
    super(CMTFTrainer,self).__init__(dataset)

    self.param['drug_feat'],self.param['cl_feat'] = read_aux(dataset,pca = False,norm = True)

  def set_trial(self,trial):
    super(CMTFTrainer,self).set_trial(trial)
 
   # Parameters
    self.hparam['aux_weight'] = trial.suggest_float('aux_weight',0,1,step = 0.01) 


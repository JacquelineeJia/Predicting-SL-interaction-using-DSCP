# Local Imports
from trainers.tf_trainer import TFTrainer

# Package Imports
from abc import ABC


class RankTrainer(TFTrainer,ABC):
   
  def __init__(self,dataset):
    super(RankTrainer,self).__init__(dataset)
    self.param['shape'] = self.shape
 
  def set_trial(self,trial):
    super(RankTrainer,self).set_trial(trial)
 
    # Parameters

    self.hparam['rank'] = trial.suggest_int('rank',1,256)


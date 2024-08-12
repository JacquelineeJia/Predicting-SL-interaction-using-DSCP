# Local Imports
from models import DeepSynergy
from trainers.tf_trainer import TFTrainer
from util.data_loader import read_aux

class DeepSynergyTrainer(TFTrainer):

  model_class = DeepSynergy
  nn_choices = [[8192,8192],[4096, 4096], [2048, 2048], [8192, 4096], [4096, 2048], [4096, 4096, 4096], [2048, 2048, 2048], [4096, 2048, 1024], [8192, 4096, 2048]] 

  def __init__(self,dataset):
    super(DeepSynergyTrainer,self).__init__(dataset)

    self.param['drug_feat'],self.param['cl_feat'] = read_aux(dataset,pca = False,norm = True)
    

  def set_trial(self,trial):
    super(DeepSynergyTrainer,self).set_trial(trial)

    self.hparam['nn'] = trial.suggest_categorical('nn',self.nn_choices)
    self.hparam['hidden_do'] = trial.suggest_float('hidden_do',0,0.5,step = 0.1)
    self.hparam['out_do'] = trial.suggest_float('out_do',0,0.2,step = 0.1) 
    self.hparam['l1_strength'] = trial.suggest_float('l1_strength', 1e-10, 1e-3, log=True)
    self.hparam['l2_strength'] = trial.suggest_float('l2_strength', 1e-10, 1e-3, log=True) 

'''
This algorithm modifies algorithms to apply pca to their feature matrics. The number of PC's is optimized as a hyperparameter.
'''

# Local Imports
from trainers.deepsynergy_trainer import DeepSynergyTrainer
from trainers.cmtf_trainer import CMTFTrainer
from trainers.dtf_trainer import DTFTrainer
from trainers.svm_trainer import SVMTrainer
from util.data_loader import read_aux

# Package Imports
import numpy as np


def pca(c):

  def set_trial(self,trial):
    super(c,self).set_trial(trial)
    self.param['drug_feat'],self.param['cl_feat'] = read_aux(self.dataset,
                                                             pca=True,
                                                             norm = True,
                                                             num_drug_pc = trial.suggest_int('num_drug_pc',1,self.shape[0]),
                                                             num_cl_pc = trial.suggest_int('num_cl_pc',1,self.shape[-1])) 
  
  c.set_trial = set_trial

  def set_hparam(self,path):
    super(c,self).set_hparam(path)
    self.param['drug_feat'],self.param['cl_feat'] = read_aux(self.dataset,
                                                             pca=True,
                                                             norm = True,
                                                             num_drug_pc = self.hparam['num_drug_pc'],
                                                             num_cl_pc = self.hparam['num_cl_pc']) 
 
    del self.hparam['num_cl_pc'] 
    del self.hparam['num_drug_pc'] 

  c.set_hparam = set_hparam
  
  return c    

@pca
class DeepSynergyPCATrainer(DeepSynergyTrainer):
  pass

@pca
class CMTFPCATrainer(CMTFTrainer):
  pass

@pca
class DTFPCATrainer(DTFTrainer):
  pass
@pca
class SVMPCATrainer(SVMTrainer):
  pass

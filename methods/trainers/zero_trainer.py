'''
This file modifies algorithms to replace their inptu features with zeros
'''

# Local Imports
from trainers.deepsynergy_trainer import DeepSynergyTrainer
from trainers.cmtf_trainer import CMTFTrainer
from trainers.dtf_trainer import DTFTrainer

# Package Imports
import numpy as np


def zero(c):
  def load_data(self,split):
    super(c,self).load_data(split)
    self.param['drug_feat'] = np.zeros_like(self.param['drug_feat'])   
    self.param['cl_feat'] = np.zeros_like(self.param['cl_feat'])   
  c.load_data = load_data
  return c    

@zero
class DeepSynergyZeroTrainer(DeepSynergyTrainer):
  pass

@zero
class CMTFZeroTrainer(CMTFTrainer):
  pass

@zero
class DTFZeroTrainer(DTFTrainer):
  pass

'''
This file modifies alogirthm to use features matrices filled with ones
'''

# Local Imports
from trainers.deepsynergy_trainer import DeepSynergyTrainer
from trainers.cmtf_trainer import CMTFTrainer
from trainers.dtf_trainer import DTFTrainer

# Package Imports
import numpy as np


def ones(c):
  def load_data(self,split):
    super(c,self).load_data(split)
    self.param['drug_feat'] = np.ones_like(self.param['drug_feat'])   
    self.param['cl_feat'] = np.ones_like(self.param['cl_feat'])   
  c.load_data = load_data
  return c    

@ones
class DeepSynergyOnesTrainer(DeepSynergyTrainer):
  pass

@ones
class CMTFOnesTrainer(CMTFTrainer):
  pass

@ones
class DTFOnesTrainer(DTFTrainer):
  pass

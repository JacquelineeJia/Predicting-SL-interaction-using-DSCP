'''
This file modifies algorithms so that they use randomized input features
'''

# Local Imports
from trainers.deepsynergy_trainer import DeepSynergyTrainer
from trainers.cmtf_trainer import CMTFTrainer
from trainers.dtf_trainer import DTFTrainer

# Package Imports
import numpy as np


def random(c):
  def __init__(self,split):
    super(c,self).__init__(split)
    self.param['drug_feat'] = np.random.normal(size = self.param['drug_feat'].shape)   
    self.param['cl_feat'] = np.random.normal(size = self.param['cl_feat'].shape)   
  c.__init__ = __init__
  return c    


@random
class DeepSynergyRandTrainer(DeepSynergyTrainer):
  pass

@random
class CMTFRandTrainer(CMTFTrainer):
  pass

@random
class DTFRandTrainer(DTFTrainer):
  pass


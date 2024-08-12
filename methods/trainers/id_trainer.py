'''
This file modifies algorithms to use identity matrix data
'''

# Local Imports
from trainers.deepsynergy_trainer import DeepSynergyTrainer
from trainers.cmtf_trainer import CMTFTrainer
from trainers.dtf_trainer import DTFTrainer
from trainers.svm_trainer import SVMTrainer

# Package Imports
import numpy as np


def id(c):
  def __init__(self,split):
    super(c,self).__init__(split)
    self.param['drug_feat'] = np.eye(self.shape[0])   
    self.param['cl_feat'] = np.eye(self.shape[-1])   
  c.__init__ = __init__
  return c    

@id
class SVMIDTrainer(SVMTrainer):
  pass

@id
class DeepSynergyIDTrainer(DeepSynergyTrainer):
  pass

@id
class CMTFIDTrainer(CMTFTrainer):
  pass

@id
class DTFIDTrainer(DTFTrainer):
  pass

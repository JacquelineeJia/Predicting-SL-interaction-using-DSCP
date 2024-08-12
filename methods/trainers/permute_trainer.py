'''
This file modifies algorithms so that they permute their input data
'''

# Local Imports
from trainers.dtf_trainer import DTFTrainer
from trainers.dscp_trainer import DSCPTrainer
from trainers.fuse_trainer import FuseTrainer
from trainers.prodeepsyn_trainer import ProDeepSynTrainer

# Package Imports
import numpy as np


def permute(c):

  def load_data(self,split,permute = True):
    super(c,self).load_data(split,permute)

  c.load_data = load_data
  return c    

@permute
class ProDeepSynPermuteTrainer(ProDeepSynTrainer):
  pass

@permute
class DSCPPermuteTrainer(DSCPTrainer):
  pass

@permute
class DTFPermuteTrainer(DTFTrainer):
  pass

@permute
class FusePermuteTrainer(FuseTrainer):
  pass

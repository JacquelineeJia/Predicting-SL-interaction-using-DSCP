# Local Imports
from models import AvgLODO
from trainers.tf_trainer import TFTrainer
from util.data_loader import read_aux

class AvgLODOTrainer(TFTrainer):

  model_class = AvgLODO

  def __init__(self,dataset):
    super(AvgLODOTrainer,self).__init__(dataset)
    self.param['shape'] = self.shape

  def load_data(self,split):
    super(AvgLODOTrainer,self).load_data(split)

    self.param['train_set'] = self.X_train 
    self.param['train_label'] = self.y_train
 

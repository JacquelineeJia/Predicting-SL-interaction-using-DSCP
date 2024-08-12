# Local Imports
from models import Avg
from trainers.tf_trainer import TFTrainer
from util.data_loader import read_aux

class AvgTrainer(TFTrainer):

  model_class = Avg

  def __init__(self,dataset):
    super(AvgTrainer,self).__init__(dataset)
    self.param['shape'] = self.shape

  def load_data(self,split):
    super(AvgTrainer,self).load_data(split)

    self.param['train_set'] = self.X_train 
    self.param['train_label'] = self.y_train
 

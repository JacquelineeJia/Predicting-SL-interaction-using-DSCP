# Local Imports
from trainers.rank_trainer import RankTrainer
from models import CMTF

class CPWoptTrainer(RankTrainer):
  
  model_class = CMTF
  stop_patience = 100
  lr_patience = 50

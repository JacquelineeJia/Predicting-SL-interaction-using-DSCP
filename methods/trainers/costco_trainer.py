# Local Imports
from models.costco import CoSTCo
from trainers.rank_trainer import RankTrainer

class CoSTCoTrainer(RankTrainer):
  
  model_class = CoSTCo
  
  def set_trial(self,trial):
    super(CoSTCoTrainer,self).set_trial(trial)
    self.hparam['reg'] = trial.suggest_float('reg',0,1)
    self.hparam['d1'] = trial.suggest_float('d1',0,0.5,step=0.1) 
    self.hparam['d2'] = trial.suggest_float('d2',0,0.5,step=0.1) 
    self.hparam['d3'] = trial.suggest_float('d3',0,0.5,step=0.1) 


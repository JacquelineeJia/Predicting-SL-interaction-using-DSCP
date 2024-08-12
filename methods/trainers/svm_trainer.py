from trainers.sk_trainer import SKTrainer
from sklearn.svm import SVR

class SVMTrainer(SKTrainer):

  model_class = SVR

  def set_trial(self,trial):
    super(SVMTrainer,self).set_trial(trial)

    self.hparam['kernel'] = trial.suggest_categorical('kernel',['linear','poly','rbf','sigmoid'])  
    self.hparam['degree'] = trial.suggest_int('degree',1,10)
    self.hparam['C'] = trial.suggest_float('C',0.01,10)


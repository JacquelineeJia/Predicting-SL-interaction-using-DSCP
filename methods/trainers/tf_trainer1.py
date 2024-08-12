# Local Imports
from trainers.base_trainer1 import Trainer
from util.met import tf_r2
from util.data_loader import prep_dataset

# Package Imports
import tensorflow.keras as tfk
from abc import ABC
import tensorflow as tf
import gc

class TFTrainer(Trainer,ABC):
  
  stop_patience = 15
  lr_patience = 10

  def __init__(self,dataset):
    super(TFTrainer,self).__init__(dataset)


    self.early_stopping = tfk.callbacks.EarlyStopping(monitor='val_loss',mode = 'min', 
                                                   patience=self.stop_patience,
                                                   min_delta = 0.001,
                                                   restore_best_weights = True)
  
    self.reduce_lr = tfk.callbacks.ReduceLROnPlateau(
      			    monitor='loss',mode='min',
  			    factor=0.5,
  			    patience=self.lr_patience,
  			    min_delta=0.001,
  			)
  
  def fit_model(self,model):
    opt = tfk.optimizers.Adam(learning_rate=self.LR) 
    model.compile(loss = 'binary_crossentropy',optimizer = opt,metrics = [tf_r2])
      
    model.fit(x = self.X_train,y = self.y_train,
              validation_data = (self.X_val,self.y_val),
              epochs = self.EPOCHS,
              verbose = self.VERBOSE,
              batch_size = 2**self.bs,
              callbacks = [self.early_stopping,self.reduce_lr])

  def pred_model(self,model,test = True):
    if test:
      X = self.X_test
      y = self.y_test
    else:
      X = self.X_train
      print("x train shape in tf :", X.shape)
      y = self.y_train
    y_test_pred = model.predict(X,batch_size = 2**self.bs,verbose = self.VERBOSE)
    y_pred = y_test_pred*self.std + self.mean
    y_true = y*self.std + self.mean

    return y_true,y_pred
 
  def train(self,split,prune = True,return_model = False):

    self.load_data(split)
    model = self.create_model()    
    self.fit_model(model)
    y_true_split,y_pred_split = self.pred_model(model)
    y_true_train,y_pred_train = self.pred_model(model,test=False)
    
    if prune:
      self.prune(split,y_true_split,y_pred_split)  

    return y_true_split,y_pred_split,y_true_train,y_pred_train
  
  def create_model(self):
    gc.collect()
    tfk.backend.clear_session() 
    return self.model_class(**self.param,**self.hparam)
  

  def save_weights(self,path):
    self.model.save_weights(path)

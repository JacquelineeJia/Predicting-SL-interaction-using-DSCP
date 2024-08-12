# Local Imports
from trainers.base_trainer import Trainer
from util.rocauc import calculate_BA
from util.data_loader import prep_dataset


from imblearn.over_sampling import SMOTE
# Package Imports
import tensorflow.keras as tfk
from abc import ABC
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import numpy as np

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
    print("Target variable unique values:", np.unique(self.y_train))
    print("Target variable data type:", self.y_train.dtype)


    # Initialize SMOTE
    smote = SMOTE(random_state=42)
    
    # Resample the training data
    X_train_resampled, y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
    
    opt = tfk.optimizers.Adam(learning_rate=self.LR) 
    
    ### change mse to 'binary_crossentropy'
    ##### change the metrics = [youdens_j_statistic] instead of tf_r2
    model.compile(loss = 'binary_crossentropy',optimizer = opt, metrics= [calculate_BA])
    history = model.fit(x = X_train_resampled ,y = y_train_resampled,
              validation_data = (self.X_val,self.y_val),
              epochs = self.EPOCHS,
              verbose = self.VERBOSE,
              batch_size = 2**self.bs,
              callbacks = [self.early_stopping,self.reduce_lr])
    return history

  def pred_model(self,model,test = True):
    if test:
      X = self.X_test
      y = self.y_test
      print("pred_model y_test shape:", y.shape)
      print("pred_model x_test shape:", X.shape)
    else:
      X = self.X_train
      y = self.y_train
      print("pred_model y_train shape:", y.shape)
      print("pred_model x_train shape:", X.shape)
    #y_test_pred = model.predict(X,batch_size = 2**self.bs,verbose = self.VERBOSE)
        
    #####
    y_pred_prob = model.predict(X, batch_size=2**self.bs, verbose=self.VERBOSE)
    threshold = 0.21  # Define your threshold
    y_pred = (y_pred_prob > threshold).astype(int)  # Apply threshold
    y_true = y
    print("pred_model y_pred:", y_pred.shape)
    print("pred_model y_true:", y_true.shape)
    return y_true,y_pred
 
  def train(self,split,prune = True,return_model = False, test=False):

    self.load_data(split)
    model = self.create_model()    
    history = self.fit_model(model)
    print("history was generated from fit_model")
    print(history)
    if not test:
        self.plot_loss(history,self.trial.number)
    print("start prediction model")
    y_true_split,y_pred_split = self.pred_model(model)
    print("prediction success")
    print("y_true split in train func:", y_true_split.shape)
    print("y_pred split in train func:", y_pred_split.shape)
    y_true_train,y_pred_train = self.pred_model(model,test=False)
    print("y_true train in train func:", y_true_train.shape)
    print("y_pred train in train func:", y_pred_train.shape)
    
    if prune:
      self.prune(split,y_true_split,y_pred_split)  
      print("finish pruning----------------")
      print("y_true_train.shape, y_pred_train.shape")
      print(y_true_train.shape, y_pred_train.shape)
    return y_true_split,y_pred_split,y_true_train,y_pred_train
  
  def create_model(self):
    gc.collect()
    tfk.backend.clear_session() 
    return self.model_class(**self.param,**self.hparam)
  

  def save_weights(self,path):
    self.model.save_weights(path)
 
  def plot_loss(self,history, trial_num):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(10,5))
    plt.plot(epochs, loss,'r',label='Training Loss')
    plt.plot(epochs, val_loss,'b',label='Validation Loss')
    plt.title(f'Train and Validate Loss - Trial {trial_num}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'./loss_plots/trial_{trial_num}_loss.png')
    plt.close()

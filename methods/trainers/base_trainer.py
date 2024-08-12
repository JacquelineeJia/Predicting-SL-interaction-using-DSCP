'''
This is the base class for the trainer
'''

# Local Imports
from util.data_loader import load_syn_data,load_syn_data_lodo
from util.data_loader import prep_dataset,prep_lodo

# Package Imports
from abc import ABC, abstractmethod
from sklearn.metrics import r2_score
import optuna
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf

class Trainer(ABC):

  EPOCHS = 100
  VERBOSE = 1
  LR = 3e-4
  BS = (9,10)
 

  # load the dataset
  def __init__(self,dataset):
    
    if dataset.split('_')[-1] == 'lodo': 
      self.splits,self.idx,self.shape,self.thresh = load_syn_data_lodo(dataset)
    else:
      print('load_syn_data initializing')
      self.splits,self.shape,self.thresh = load_syn_data(dataset)
      print('finished loading.........')
    self.dataset = dataset
    self.param = {}

  # Define the trial pruning threshold
  def prune(self,split,y_true,y_pred):
    r2 = r2_score(y_true,y_pred)
    self.trial.report(r2,split) 
    if self.trial.should_prune():
      raise optuna.TrialPruned()

  # Stores the optuna trial as a instance variable
  def set_trial(self,trial):
 
    self.trial = trial 
    
    # Parameters
    self.hparam = {}
    self.bs = trial.suggest_int('bs',self.BS[0],self.BS[1]) 

  # Execute a training of the algorithm and test on the testing set
  def test(self): 
    return self.train('test',prune = False,return_model = True, test=True)

  # Set hyperparameters from an optimal hyperparamater .npy file
  def set_hparam(self,path):
    self.hparam = np.load(path,allow_pickle = True).item()
    self.bs = self.hparam['bs']
    del self.hparam['bs'] 

  # Drug Drug synergy is symmetric, this funciton adds both sides of the symmetry to the data
  def stack_x(x):
    return np.concatenate([x,np.stack([x[:,1],x[:,0],x[:,2]],axis = 1)],axis = 0)

  # Eliminate the redundent data due to symmentry
  def unstack_x(x):
    return x[:x.shape[0]//2]

  # Repeat the labels
  def stack_y(y):
    return np.concatenate([y,y],axis = 0)
 
  # Remove repeated the labels
  def unstack_y(y):
    return y[:y.shape[0]//2]

  # Split the dataset into train, test and validation splits
  def load_data(self,split,permute = False):
    if self.dataset.split('_')[-1] == 'lodo':
      self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test,self.mean,self.std = prep_lodo(self.splits,split,self.dataset,self.idx)
    else:
      self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test,self.mean,self.std = prep_dataset(self.splits,split)
    self.X_train = Trainer.stack_x(self.X_train)
    self.X_val   = Trainer.stack_x(self.X_val)
    self.X_test  = Trainer.stack_x(self.X_test)
    self.y_train = Trainer.stack_y(self.y_train)
    self.y_val   = Trainer.stack_y(self.y_val)
    self.y_test  = Trainer.stack_y(self.y_test)
    print("y_test in load data")
    print(self.y_test)
    print("y_train shape in load data")
    print(self.y_train.shape)

    if permute:
      drug_permute = np.arange(self.shape[0]) 
      cl_permute =   np.arange(self.shape[-1])
      np.random.shuffle(drug_permute)
      np.random.shuffle(cl_permute)
	     
      self.X_train[:,0] = drug_permute[self.X_train[:,0]]
      self.X_train[:,1] = drug_permute[self.X_train[:,1]]
      self.X_train[:,2] = cl_permute[self.X_train[:,2]]
    
      self.X_test[:,0] = drug_permute[self.X_test[:,0]]
      self.X_test[:,1] = drug_permute[self.X_test[:,1]]
      self.X_test[:,2] = cl_permute[self.X_test[:,2]]
	    
      self.X_val[:,0] = drug_permute[self.X_val[:,0]]
      self.X_val[:,1] = drug_permute[self.X_val[:,1]]
      self.X_val[:,2] = cl_permute[self.X_val[:,2]]
      print("======= in if permute statement=======")
      print("X train shape:", X_train.shape)
      print("X test shape:", X_test.shape)
      print("=========permute done======")
  # train the algorithm
  @abstractmethod
  def train(self,split,prune = True,test=False):
    print("we are passing through train")
    pass
    print("train has been passed")

  # save the trained alogirthms weights
  @abstractmethod
  def save_weights(self,path):
    pass

  # Execute a full cross validation sequence on the training split
  def predict(self):
 
    if self.dataset.split('_')[0] == 'genecomb':
      
      y_true = []
      y_pred = []
      for i in range(5):
        y_true_split,y_pred_split,_,_ = self.train(i)
        print("========looping train(i)=======")
        y_true.append(y_true_split)
        y_pred.append(y_pred_split)
        print("y_true,y_pred, i")
        print(y_true,y_pred, i)
      y_true = np.concatenate(y_true,axis = 0)
      y_pred = np.concatenate(y_pred,axis = 0)
      print("========predict model calling=========")
      print("y_true:")
      print(y_true.shape)
      print("---y_pred------")
      print(y_pred.shape)
    elif self.dataset in ['almanac' ,'merick' ,'drugcomb','drugcomb_cc','almanac_loclo']:
     y_true,y_pred,_,_ = self.train(0)
     print(y_true,y_pred,_,_)
    print("y_pred shape", y_pred.shape)
    print("y_true shape",y_true.shape)
    return y_true,y_pred
    
  #######output  
  def calculate_rocauc(self,y_true, y_pred):
    # Convert probabilities to binary predictions if necessary
    # Assuming y_pred is binary
    roc_auc = roc_auc_score(y_true, y_pred)
    return roc_auc

  # Return the coeffecient of correlation for a trial
  def get_objective(self):
    y_true,y_pred = self.predict()
    y_pred = y_pred.squeeze()
    y_pred_binary = (y_pred > 0.2).astype(int)  # Apply threshold to get binary predictions
    y_true= (y_true > 0.2).astype(int)
    print("Shape of y_true:", np.shape(y_true))
    print("Shape of y_pred_binary:", np.shape(y_pred_binary))
    print("Unique predictions:", np.unique(y_pred_binary))  # Debug: Check prediction diversity
    print("Unique true labels:", np.unique(y_true))  # Debug: Check true label diversity
    

    # Convert numpy arrays to tensors
    y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.int32)
    y_pred_binary_tf = tf.convert_to_tensor(y_pred_binary, dtype=tf.int32)

    # Use TensorFlow to calculate confusion matrix elements
    cm = tf.math.confusion_matrix(y_true_tf, y_pred_binary_tf)
    # Ensure the confusion matrix is 2x2
    if cm.shape[0] == 2:
        # Extract true positives, true negatives, false positives, false negatives
        TN, FP, FN, TP = tf.reshape(cm, [-1])
    else:
        print("Confusion matrix shape:", cm.shape, "Expected a binary classification problem.")
        return 0

    # Calculate sensitivity (recall) and specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # Calculate balanced accuracy
    balanced_accuracy = (sensitivity + specificity) / 2

    print(f"Balanced Accuracy: {balanced_accuracy.numpy()}")
    return balanced_accuracy.numpy()
   # return r2_score(y_true,y_pred)

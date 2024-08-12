'''
This file trains an algorithm on the training set and generates predictions on the testing set.
Because it uses the optimal hyperparameters, it is neccessary to run opt.py for an algorithm first.
The syntax is as follows:
python test.py [dataset] [algorithm] [gpu]
For example, python test.py merick dscp 0 would generate 10 test set predictions for the DSCP algorithm on the merick and co compound screen using gpu number 0
The results are saved to the results folder.
'''

# Local Imports
from util.param import PATH
from util.send_email import send_mail
from trainers import trainer_lookup

# Package Imports
import sys
import os
import optuna
import numpy as np
from scipy import stats

# restrict the gpu to the specified gpu
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[3]

# Read the dataset and method
dataset = sys.argv[1]
method = sys.argv[2]

# Get the appropiate trainer and initilize it with the dataset
trainer = trainer_lookup[method](dataset)

# Reads the optimal hypermaters from the results folder
params_path = PATH+"results/{}/{}/optimal_params.npy".format(dataset,method)
 

# The algorithm is trained ten times and evaluated on the testing set ten times. These lists store the output of each test
y_test_pred_out = []
y_test_out= []
y_train_pred_out = []
y_train_out = []

try:
  print("trying our try/catch block")
  for _ in range(10):
    
    print("trainer about to set hparam")
    # Set the optimal hyperparameters
    trainer.set_hparam(params_path)
    print("trainer successfully set hparams")
  
    # Get the predictions and ground truth for the train and test set using the optimal hyperparameters
    y_test_true,y_test_pred,y_train_true,y_train_pred = trainer.test()
    
    print("y_test_true: ", y_test_true)

    # Add these to the lists
    y_test_pred_out.append(y_test_pred)
    y_test_out.append(y_test_true)
    y_train_pred_out.append(y_train_pred)
    y_train_out.append(y_train_true)

  print(f'y_test_pred_out length: {len(y_test_pred_out)}')
  print(f'y_test_out length: {len(y_test_out)}')
  print(f'y_train_pred_out length: {len(y_train_pred_out)}')
  print(f'y_train_out length: {len(y_train_out)}')

except Exception as e:
 # send_mail('test.py',method,dataset,error = True)
 # raise e
 pass
  

# Write the predictions and ground truth the the results folder
print(f'y_test_pred_out length: {len(y_test_pred_out)}')
y_test_pred_out = np.stack(y_test_pred_out,axis = 0)
y_test_out = np.stack(y_test_out,axis = 0)
y_train_pred_out = np.stack(y_train_pred_out,axis = 0)
y_train_out = np.stack(y_train_out,axis = 0)

try:
  np.save(PATH+'results/{}/{}/y_test_pred.npy'.format(dataset,method),y_test_pred_out)
  np.save(PATH+'results/{}/{}/y_test.npy'.format(dataset,method),y_test_out)
  np.save(PATH+'results/{}/{}/y_train_pred.npy'.format(dataset,method),y_train_pred_out)
  np.save(PATH+'results/{}/{}/y_train.npy'.format(dataset,method),y_train_out)
  trainer.save_weights(PATH+'results/{}/{}/weights'.format(dataset,method))
except:
  os.mkdir(PATH+"results/{}/{}".format(dataset,method))
  np.save(PATH+'results/{}/{}/y_test_pred.npy'.format(dataset,method),y_test_pred_out)
  np.save(PATH+'results/{}/{}/y_test.npy'.format(dataset,method),y_test_out)
  np.save(PATH+'results/{}/{}/y_train_pred.npy'.format(dataset,method),y_train_pred_out)
  np.save(PATH+'results/{}/{}/y_train.npy'.format(dataset,method),y_train_out)
  trainer.save_weights(PATH+'results/{}/{}/weights'.format(dataset,method))

# Send an email to indicate that the testing script executed successfully
#send_mail('test.py',method,dataset)

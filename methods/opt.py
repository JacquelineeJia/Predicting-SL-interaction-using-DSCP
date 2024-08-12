'''
This file executes the hyperparameter optimization for a particular algorithm. 
It should be ran with the following inputs:
python opt.py [dataset] [algoritm] [gpu]
For example, python opt.py merick dscp 0 would run a hyperparameter optimization for dscp on the merick and co dataset using gpu number 0.
The results are saved to the results folder.
'''

# Local Imports
from util.param import PATH
from util.send_email import send_mail
from trainers import trainer_lookup

# Package Imports
import os
import optuna
import numpy as np
import sys
import tensorflow as tf
import traceback

# restrict the gpu to the specified gpu
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[3]

# Read the dataset and method
dataset = sys.argv[1]
method = sys.argv[2]

# Get the appropiate trainer and initilize it with the dataset
try:
    trainer = trainer_lookup[method](dataset)
    print("look up success")
except Exception as e:
    print(e)
    traceback.print_exc()



def objective(trial):

  # Add's the hyperparameter ranges to the trial
  print("setting trails")
  trainer.set_trial(trial)
  print("done setting ........")

  # Returns the objective for the given hyperparameters
  return trainer.get_objective()


# Creates an optuna study to maximize the objective of our trial 
study = optuna.create_study(direction='maximize',pruner = None)


# OPtimize the study for 100 trials, send an error email if it fails
try:
  study.optimize(objective, n_trials=5)
except Exception as e: 
  send_mail('opt.py',method,dataset,error = True)
  raise e

# Write the results to the results folder, creating the appropiate folder if it does not exist
try:
  np.save(PATH+"results/{}/{}/optimal_params.npy".format(dataset,method),study.best_params)
  study.trials_dataframe().to_csv(PATH+"results/{}/{}/trials_df.csv".format(dataset,method))
except:
  os.mkdir(PATH+"results/{}/{}".format(dataset,method))
  np.save(PATH+"results/{}/{}/optimal_params.npy".format(dataset,method),study.best_params)
  study.trials_dataframe().to_csv(PATH+"results/{}/{}/trials_df.csv".format(dataset,method))

# Send an email to indicate that the hyperparameter optimization succesfully completed
#send_mail('opt.py',method,dataset)

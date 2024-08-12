# Local Imports
from util.param import PATH

# Package Imports
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA


# This function loads a synthetic dataset using the ldo cross validation strategy
def load_syn_data_lodo(dataset):
  if dataset.split('_')[0] ==  'merick':
    shape = [38,38,39]
    thresh = 30
  elif dataset.split('_')[0] == 'almanac':
    shape = [102,102,60]
    thresh = 10
  elif dataset.split('_')[0] == 'drugcomb':
    shape = [3040,3040,81]
    thresh = 10
  elif dataset.split('_')[0] == 'genecomb':
    shape = [2075, 2075,8]
    thresh = 5
  splits = [{'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]}]


  for s in range(6): 
    splits[s]['X'] = np.load(PATH+'data/{}/s{}.npy'.format(dataset,s)).astype('int32')

  idx = np.load(PATH + 'data/{}/idx.npy'.format(dataset)) 

  return splits,idx,shape,thresh
  


# This function loads a synthetic dataset using the ldco cross validation strategy
def load_syn_data(dataset):
  if dataset in ['merick','merick_loclo']:
    shape = [38,38,39]
    thresh = 30
  elif dataset in ['almanac','almanac_loclo']:
    shape = [102,102,60]
    thresh = 10
  elif dataset in ['drugcomb' , 'drugcomb_cc']:
    shape = [3040,3040,81]
    thresh = 10
  elif dataset == 'genecomb':
    shape = [2075, 2075, 8]
    thresh = 5
    print(shape,thresh)
    print("==============================")
  
  splits = [{'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]}]


  for s in range(6): 
    print("starting split")
    splits[s]['X'] = np.load(PATH+'data/{}/s{}.npy'.format(dataset,s))[:,0:-1].astype('int32')
    splits[s]['y'] = np.load(PATH+'data/{}/s{}.npy'.format(dataset,s))[:,-1]    
 
  return splits,shape,thresh


# This function prepares a ldo dataset using a particular split of the data
def prep_lodo(splits,i,dataset,idx):
      # Prep Data
      X_train = []
      y_train = []
      X_test = []
      y_test = []
      X_val = []
      y_val = []
 

      train_drugs = []
      test_drugs = []
      val_drugs = [] 
      if i == 'test':
        test_drugs = splits[-1]['X']
        val_drugs = splits[0]['X']
        train_drugs = [x['X'] for x in splits[1:-1]]
      else:
        for s in range(5):
          if s == i:
            val_drugs = splits[s]['X']
            test_drugs = splits[s]['X']
          else:
            train_drugs.append(splits[s]['X'])
      train_drugs = np.concatenate(train_drugs,axis=0)

      for row in idx:

        train_count = 0
        test_count = 0
        val_count = 0

        train_count += row[0] in train_drugs
        test_count += row[0] in test_drugs
        val_count += row[0] in val_drugs
        
        train_count += row[1] in train_drugs
        test_count += row[1] in test_drugs
        val_count += row[1] in val_drugs

        if train_count == 2:
          X_train.append(row[:-1])
          y_train.append(row[-1])
        if test_count == 2:
          X_test.append(row[:-1])
          y_test.append(row[-1])
        if val_count == 2:
          X_val.append(row[:-1])
          y_val.append(row[-1])
        if train_count == 1 and test_count == 1:
          X_test.append(row[:-1])
          y_test.append(row[-1])
        if train_count == 1 and val_count == 1:
          X_val.append(row[:-1])
          y_val.append(row[-1])
      X_train = np.stack(X_train,axis=0)
      y_train = np.array(y_train)
      X_test = np.stack(X_test,axis=0)
      y_test = np.array(y_test)
      X_val = np.stack(X_val,axis=0)
      y_val = np.array(y_val)
  
      print(X_train.shape)
      print(X_test.shape)
      print(X_val.shape)
  
      mean = np.mean(y_train)
      std = np.std(y_train)
  
      y_test = (y_test - mean)/std
      y_val = (y_val - mean)/std
      y_train = (y_train - mean)/std


      return X_train.astype('int32'),y_train,X_val.astype('int32'),y_val,X_test.astype('int32'),y_test,mean,std 

# This function preps the dataaset using a particular split of the data
def prep_dataset(splits,i):
      # Prep Data
      X_train = []
      y_train = []
      X_test = []
      y_test = []
  
      if i == 'test':
        X_test = splits[-1]['X']
        y_test = splits[-1]['y']
        X_val = splits[0]['X']
        y_val = splits[0]['y']
        X_train = [x['X'] for x in splits[1:-1]]
        y_train = [x['y'] for x in splits[1:-1]]
      else:
        for s in range(5):
          if s == i:
            X_val = splits[s]['X']
            y_val = splits[s]['y']
            X_test = splits[s]['X']
            y_test = splits[s]['y']
          else:
            X_train.append(splits[s]['X'])
            print("X_train in loop shape: ", len(X_train))
            y_train.append(splits[s]['y'])
      X_train = np.concatenate(X_train,axis=0)
      y_train = np.concatenate(y_train,axis=0)

        
      mean = np.mean(y_train)
      std = np.std(y_train)
  

      print("y_test shape:", y_test.shape)
      y_test = (y_test - mean)/std
      y_val = (y_val - mean)/std
    
    
      print("X_train shape in data loader===========")
      print(X_train.shape)
      return X_train.astype('int32'),y_train,X_val.astype('int32'),y_val,X_test.astype('int32'),y_test,mean,std 

# reads the drug and cell line similarity matrices
def read_sim(dataset):
   
    df = np.load(PATH+'data/{}/drug_sim.npy'.format(dataset)).astype('float32')
    cl = np.load(PATH+'data/{}/cline_sim.npy'.format(dataset)).astype('float32')
      
    return df,cl

# Reads the auxillary data for the drug and cell line
def read_aux(dataset,pca = False,norm = False,num_drug_pc = None,num_cl_pc = None,cl_aux_type = None,drug_aux_type = None):
   
    if not drug_aux_type:
      df = np.load(PATH+'data/{}/drug_features.npy'.format(dataset)).astype('float32')
    else:
      df = np.load(PATH+'data/{}/{}_drug_features.npy'.format(dataset,drug_aux_type)).astype('float32')

    if not cl_aux_type:
      cl = np.load(PATH+'data/{}/cl_features.npy'.format(dataset)).astype('float32')
    else:
      cl = np.load(PATH+'data/{}/{}_cl_features.npy'.format(dataset,cl_aux_type)).astype('float32')


    if pca:
        df = PCA(n_components = min(num_drug_pc,df.shape[1])).fit_transform(df)
        cl = PCA(n_components = min(num_cl_pc,df.shape[1])).fit_transform(cl)
    if norm:
        df -= df.mean(axis = 0)
        df = df[:,df.std(axis = 0) > 1e-5]
        df /= df.std(axis = 0)
        df = df[:,df.std(axis = 0) > 1e-5]
        cl -= cl.mean(axis = 0)
        cl = cl[:,cl.std(axis = 0) > 1e-5]
        cl /= cl.std(axis = 0)
        cl = cl[:,cl.std(axis = 0) > 1e-5]
      
    return df,cl

# Reads the drug synergy tensor
def read_tensor(dataset):
  return np.load(PATH+'data/{}/dds.npy'.format(dataset))

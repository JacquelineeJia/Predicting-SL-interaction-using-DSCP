# Local Imports
from models import GraphEmbed
from trainers.torch_trainer import TorchTrainer
from util.data_loader import read_aux
from rdkit import Chem
import pandas as pd
import numpy as np
import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, mean_squared_error, r2_score
from scipy.stats import pearsonr
import random
import deepchem as dc
from torch_geometric.data import Data,Batch
from sklearn.metrics import r2_score

def getData(dataset):
   
  try:
    drug_graph_data = np.load('/home/judah/dscp/data/{}/drug_graph.npy'.format(dataset),allow_pickle = True)
  except:
    drug_smiles_file = '/home/judah/dscp/data/{}/smi.npy'.format(dataset)
    drug = np.load(drug_smiles_file)
    featurizer = dc.feat.MolGraphConvFeaturizer()
    drug_graph_data = featurizer.featurize([sm for sm in drug])
    np.save('/home/judah/dscp/data/{}/drug_graph.npy'.format(dataset),drug_graph_data,allow_pickle = True)

  pyg_graph = []
  for graph in drug_graph_data:
    if not isinstance(graph,np.ndarray):
      pyg_graph.append(Data(graph.node_features,graph.edge_index))
   

  return  Batch.from_data_list(pyg_graph)


class GraphEmbedTrainer(TorchTrainer):

  model_class = GraphEmbed
  nn_choices = [[512],
                [1024],
                [2048],
                [4096],
                [512,512],
                [1024,1024],
                [2048,2048],
                [4096,4096],
                [512,512,512],
                [1024,1024,1024],
                [2048,2048,2048],
                [4096,4096,4096],
                [2048,1024,512],
                [4096,2048,1024],
                [1024,512],
                [2048,1024],
                [4096,2048]] 
  BS = (10,12)

  def __init__(self,dataset):
    super(GraphEmbedTrainer,self).__init__(dataset)

    _,self.param['cl_feat'] = read_aux(dataset,pca = False,norm = True)
    self.param['shape'] = self.shape 
    self.param['drug_graph'] = getData(dataset)
    

  def set_trial(self,trial):
    super(GraphEmbedTrainer,self).set_trial(trial)
 
    self.hparam['nn'       ] = trial.suggest_categorical('nn',self.nn_choices)
    self.hparam['hidden_do'] = trial.suggest_float('hidden_do',0.0,0.5,step = 0.1)
    self.hparam['out_do'   ] = trial.suggest_float('out_do',0.0,0.2,step = 0.1)    
    self.hparam['graph_do'] = trial.suggest_float('graph_do',0.0,0.5,step = 0.1)
    self.hparam['embed_dim'] = trial.suggest_int('embed_dim',32,256)
    self.hparam['graph_layers'] = trial.suggest_int('graph_layers',1,6)

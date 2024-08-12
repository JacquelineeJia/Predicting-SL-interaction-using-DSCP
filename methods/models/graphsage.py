'''
This file is similar to graphembed.py except that graphsage is used to embed the drug molecular graph
'''

import numpy as np
from torch_geometric.data import Batch
import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GCNConv, global_max_pool, global_mean_pool
from torch_geometric.nn.models import GCN,GraphSAGE


class DrugEmbed(nn.Module):
    def __init__(self,output,layers,dr):
        super(DrugEmbed, self).__init__()
        self.gcn = GraphSAGE(in_channels = -1,
                       hidden_channels = output,
                       num_layers = layers,
                       norm = 'layer_norm',
                       dropout = dr)
        self.norm = torch.nn.LayerNorm(output)        
        self.out = torch.nn.Linear(output,output)
        self.act = torch.nn.ReLU()
    def forward(self, atom_features, edge_index, batch):
        x = self.gcn(atom_features,edge_index)
        x = global_max_pool(x, batch)
        x = self.out(x)
        x = self.act(x)
        x = self.norm(x)
        return x


class CellEmbed(nn.Module):
    def __init__(self,shape,embed_dim):
        super(CellEmbed, self).__init__()
        self.cl_feat = torch.tensor(np.eye(shape[-1],embed_dim),dtype=torch.float,device=torch.device('cuda'))
        self.norm = torch.nn.LayerNorm(embed_dim)     
        
    def forward(self,cell_index):
        return self.norm(self.cl_feat[cell_index])

class GraphEmbed(torch.nn.Module):
  def __init__(self,
               drug_graph,
               cl_feat,
               shape,
               nn,
               out_do,
               hidden_do,
               embed_dim,
               graph_layers,
               graph_do):
    
    super(GraphEmbed,self).__init__()
   
    self.drug_graph = drug_graph
      
    self.trip_embed = []
    prev_shape = embed_dim*3
    for i in range(0,len(nn)):
      self.trip_embed.append(torch.nn.Linear(prev_shape,nn[i]))
      prev_shape=nn[i]
      self.trip_embed.append(torch.nn.ReLU())
      if i + 1 < len(nn):
        self.trip_embed.append(torch.nn.LayerNorm(nn[i]))
        self.trip_embed.append(torch.nn.Dropout(hidden_do))
      else:  
        self.trip_embed.append(torch.nn.Dropout(out_do)) 
    self.trip_embed = torch.nn.Sequential(*self.trip_embed)
 
    self.drug_embed = DrugEmbed(embed_dim,graph_layers,graph_do)
    self.cell_embed = CellEmbed(shape,embed_dim)
    
    self.trip_predict = torch.nn.Linear(prev_shape,1)
   

  def forward(self,indices):

    da = self.drug_graph

    atom_features =[]
    edge_index = []
    total_atoms = 0
    for i in range(len(da.edge_index)):
      atoms = da.x[i]
      edges = da.edge_index[i]
      num_atoms = atoms.shape[0]
      atom_features.append(atoms)
      edge_index.append(edges+total_atoms)
      total_atoms += num_atoms

    atom_features = torch.tensor(np.concatenate(atom_features,axis = 0),dtype = torch.float,device = 'cuda')
    edge_index = torch.tensor(np.concatenate(edge_index,axis = -1),device = 'cuda')
    batch = da.batch.to('cuda')
    drug_features = self.drug_embed(atom_features,edge_index,batch)
     
    da = drug_features[indices[...,0]]
    db = drug_features[indices[...,1]]
    cl = self.cell_embed(indices[...,2])
    
    trips = torch.cat([da,db,cl],axis = -1)
    trips = self.trip_embed(trips)
    x = self.trip_predict(trips)

    return x[:,0]




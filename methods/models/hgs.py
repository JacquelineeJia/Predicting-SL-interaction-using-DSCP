'''
This file attempts to implement hyphergraphsynergy. I was unable to get it to work.
'''

import numpy as np
from torch_geometric.data import Batch
import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GCNConv, global_max_pool, global_mean_pool
from torch_geometric.nn.models import GCN,GraphSAGE


class HGNN(torch.nn.Module):
    def __init__(self, X_train,y_train,shape):
        super(HGNN, self).__init__()

        NINTY_PERCENTILE = 1.2813
          
        X_train = X_train[y_train > NINTY_PERCENTILE]
        idx = np.arange(X_train.shape[0])
        X_train[:,-1] += shape[0]
        X_train = X_train.reshape(-1)
        idx = np.stack([idx,idx,idx],axis = 1).reshape(-1)
        edge = np.stack([X_train,idx],axis = 0)
        self.edge = torch.tensor(edge,dtype=torch.long,device='cuda') 

        self.conv1 = HypergraphConv(100, 256)
        self.batch1 = nn.BatchNorm1d(256)
        self.conv2 = HypergraphConv(256, 256)
        self.batch2 = nn.BatchNorm1d(256)
        self.conv3 = HypergraphConv(256, 256)
        self.act = nn.LeakyReLU()

    def forward(self, drug_feature,cl_feature):
        drug_shape = drug_feature.shape[0]
        x = torch.cat([drug_feature,cl_feature],0) 
        x = self.batch1(self.act(self.conv1(x, self.edge)))
        x = self.batch2(self.act(self.conv2(x, self.edge)))
        x = self.act(self.conv3(x, self.edge))
        return x[:drug_shape],x[drug_shape:]


class DrugEmbed(nn.Module):
    def __init__(self):
        super(DrugEmbed, self).__init__()
        self.gcn = GCN(in_channels = -1,
                       hidden_channels = 128,
                       out_channels = 100,
                       num_layers = 2,
                       norm = 'batch_norm',
                       dropout = 0.3,
                       act = 'relu')
    def forward(self, atom_features, edge_index, batch):
        x = self.gcn(atom_features,edge_index)
        x = global_max_pool(x, batch)
        return x



class CellEmbed(nn.Module):
    def __init__(self,ge_dim):
      super(CellEmbed, self).__init__()
    
      self.embed = [] 
      self.embed.append(torch.nn.Linear(ge_dim,128))
      self.embed.append(torch.nn.Tanh())
      self.embed.append(torch.nn.BatchNorm1d(128))
      self.embed.append(torch.nn.Dropout(0.3))
      self.embed.append(torch.nn.Linear(128,100))
      self.embed.append(torch.nn.ReLU())
      self.embed = torch.nn.Sequential(*self.embed)
   
    def forward(self,gene_expression):
        return self.embed(gene_expression)


class HyperGraphSynergy(torch.nn.Module):
  def __init__(self,
               drug_graph,
               cl_feat,
               drug_sim,
               cl_sim,
               shape,
               X_train,
               y_train,
               nn,
               hidden_do,
               out_do):
    
    super(HyperGraphSynergy,self).__init__()
    
    self.drug_graph = drug_graph
    self.cl_feat = torch.tensor(cl_feat,dtype=torch.float,device='cuda')
    self.shape = shape 
    da = self.drug_graph
    self.drug_sim = torch.tensor(drug_sim,dtype = torch.float,device='cuda')
    self.cl_sim = torch.tensor(cl_sim,dtype = torch.float,device='cuda')
    self.drug_sim_emb = torch.tensor(np.random.normal(size=(256,256)),dtype=torch.float,device='cuda',requires_grad = True) 
    self.cl_sim_emb = torch.tensor(np.random.normal(size=(256,256)),dtype=torch.float,device='cuda',requires_grad = True) 
    
    self.cl_id = torch.tensor(np.eye(shape[-1],100),dtype=torch.float,device='cuda') 
    self.drug_id = torch.tensor(np.eye(shape[0],100),dtype=torch.float,device='cuda') 
  
 
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

    self.atom_features = torch.tensor(np.concatenate(atom_features,axis = 0),dtype = torch.float,device = 'cuda')
    self.edge_index = torch.tensor(np.concatenate(edge_index,axis = -1),device = 'cuda')
    self.batch = da.batch.to('cuda')
     
    self.mlp = []
    prev_shape = 256*3
    for i in range(0,len(nn)):
        self.mlp.append(torch.nn.Linear(prev_shape,nn[i]))
        prev_shape=nn[i]
        self.mlp.append(torch.nn.ReLU())
        self.mlp.append(torch.nn.BatchNorm1d(prev_shape))
        self.mlp.append(torch.nn.Dropout(out_do*(1-min(1,i)) + hidden_do*(min(1,i))))
  
    self.mlp.append(torch.nn.Linear(prev_shape,1))
    self.mlp = torch.nn.Sequential(*self.mlp)
  
    self.drug_embed = DrugEmbed()
    self.cell_embed = CellEmbed(self.cl_feat.shape[-1]) 
  
    self.hgnn = HGNN(X_train,y_train,shape)
    

  def forward(self,indices):

    drug_features = self.drug_embed(atom_features = self.atom_features, 
                                    edge_index = self.edge_index, 
                                    batch = self.batch)
    cl_features = self.cell_embed(self.cl_feat)
     
    drug_features,cl_features = self.hgnn(drug_features,cl_features)
    
    drug_sim_pred = torch.sigmoid(torch.einsum('ab,bc,dc->ad',drug_features,self.drug_sim_emb,drug_features))
    cl_sim_pred = torch.sigmoid(torch.einsum('ab,bc,dc->ad',cl_features,self.cl_sim_emb,cl_features))
        
    drug_sim_loss = torch.nn.BCELoss()(drug_sim_pred,self.drug_sim)
    cl_sim_loss = torch.nn.BCELoss()(cl_sim_pred,self.cl_sim)
    sim_loss = drug_sim_loss + cl_sim_loss

    da = drug_features[indices[...,0]]
    db = drug_features[indices[...,1]]
    cl = cl_features[indices[...,2]]
     
    trips = torch.cat([da,db,cl],axis = -1)
    x = self.mlp(trips)
    
    return x[:,0],sim_loss




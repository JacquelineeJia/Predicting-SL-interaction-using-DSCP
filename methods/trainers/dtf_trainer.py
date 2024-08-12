# Local Imports
from trainers.deepsynergy_trainer import DeepSynergyTrainer
from models import CMTF

# Package Imports
import numpy as np


class DTFTrainer(DeepSynergyTrainer):

  nn_choices = [[1024, 1024, 512],
                [2048, 2048, 1024],
                [2048, 1024, 512],
                [2048, 2048, 2048],
                [1024, 1024, 1024],
                [2048, 2048],
                [1024, 1024],
                [512, 512],
                [1024, 512]]

  rank = 1000

  def __init__(self,dataset):
    # Skip DeepSynergies init func
    super(DeepSynergyTrainer,self).__init__(dataset)
 
  def load_data(self,split,permute = False):
    super(DTFTrainer,self).load_data(split,permute)

    cp_wopt = CMTF(shape = self.shape,rank = self.rank)
    
    # Compile the model here
    cp_wopt.compile(optimizer='adam', 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])
    ############

    print('CP_WOPT Feature Extraction')
    self.fit_model(cp_wopt)
    print('Feature Extraction Finished')
    df_feat = cp_wopt.C_1.weights[0].numpy()
    cl_feat = cp_wopt.C_3.weights[0].numpy()
    df_feat -= np.mean(df_feat,axis = 0)
    df_feat /= np.std(df_feat,axis = 0)
    cl_feat -= np.mean(cl_feat,axis = 0)
    cl_feat /= np.std(cl_feat,axis = 0)
    self.param['drug_feat'] = df_feat
    self.param['cl_feat'] = cl_feat
    

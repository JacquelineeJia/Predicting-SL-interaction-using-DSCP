# Local Imports
from trainers.deepsynergy_trainer import DeepSynergyTrainer
from util.data_loader import read_aux

class ProDeepSynTrainer(DeepSynergyTrainer):


  def __init__(self,dataset):
    super(DeepSynergyTrainer,self).__init__(dataset)

    self.param['drug_feat'],self.param['cl_feat'] = read_aux(dataset,pca = False,norm = True,cl_aux_type = 'prodeepsyn')
     

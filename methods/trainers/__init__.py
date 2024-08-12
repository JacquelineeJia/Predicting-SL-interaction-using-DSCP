from trainers.base_trainer import Trainer
from trainers.deepsynergy_trainer import DeepSynergyTrainer
from trainers.rand_trainer import DeepSynergyRandTrainer
from trainers.permute_trainer import ProDeepSynPermuteTrainer,DTFPermuteTrainer,DSCPPermuteTrainer,FusePermuteTrainer
from trainers.id_trainer import DeepSynergyIDTrainer
from trainers.dscp_trainer import DSCPTrainer
from trainers.fuse_trainer import FuseTrainer
from trainers.fuse_nae_trainer import FuseNAETrainer
from trainers.dtf_trainer import DTFTrainer
from trainers.cp_wopt_trainer import CPWoptTrainer
from trainers.costco_trainer import CoSTCoTrainer
from trainers.avg_trainer import AvgTrainer
from trainers.avg_lodo_trainer import AvgLODOTrainer
from trainers.hgs_trainer import HyperGraphSynergyTrainer
from trainers.prodeepsyn_trainer import ProDeepSynTrainer


trainer_lookup = {'deepsynergy'     :DeepSynergyTrainer,
                  'deepsynergyid'     :DeepSynergyIDTrainer,
                  'prodeepsyn'     :ProDeepSynTrainer,
                  'deepsynergyrand'     :DeepSynergyRandTrainer,
                  'dtf':DTFTrainer,
                  'cp_wopt':CPWoptTrainer,
                  'dscp':DSCPTrainer,
                  'costco':CoSTCoTrainer,
                  'avg':AvgTrainer,
                  'avg_lodo':AvgLODOTrainer,
                  'fuse':FuseTrainer,
                  'fuse_nae':FuseNAETrainer,
                  'hgs':HyperGraphSynergyTrainer,
                  'dtfperm':DTFPermuteTrainer,
                  'prodeepsynperm':ProDeepSynPermuteTrainer,
                  'fuseperm':FusePermuteTrainer,
                  'dscpperm':DSCPPermuteTrainer}

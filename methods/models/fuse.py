import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import numpy as np
import tensorly as tl
from models import SCP
tl.set_backend('tensorflow')

# Reconstructs the drug synergy tensor using an MLP
class TensorDecoder(tfk.Model):
  def __init__(self,
               nn,
               out_do,
               hidden_do):
    super(TensorDecoder,self).__init__()
    

    self.mlp = []
    for i in range(0,len(nn)):
      self.mlp.append(tfkl.Dense(nn[i],kernel_initializer='he_normal'))
      if i != (len(nn) -1):  
        self.mlp.append(tfkl.BatchNormalization())
      self.mlp.append(tfkl.ReLU())
      self.mlp.append(tfkl.Dropout(out_do*(1-min(1,i)) + hidden_do*(min(1,i))))
  
    self.mlp.append(tfkl.Dense(1,kernel_initializer='he_normal'))

  def call(self,x):
    

    for l in self.mlp:
      x = l(x)

    return x

# Reconstructs the gene expression and drug chemical data
class AuxDecoder(tfk.Model):
  def __init__(self,
               out_dim,
               nn,
               out_do,
               hidden_do):
    super(AuxDecoder,self).__init__()
    

    self.mlp = []
    for i in range(0,len(nn)):
      self.mlp.append(tfkl.Dense(nn[i],kernel_initializer='he_normal'))
      if i != (len(nn) -1):  
        self.mlp.append(tfkl.BatchNormalization())
      self.mlp.append(tfkl.ReLU())
      self.mlp.append(tfkl.Dropout(out_do*(1-min(1,i)) + hidden_do*(min(1,i))))
  
    self.mlp.append(tfkl.Dense(out_dim,kernel_initializer='he_normal'))

  def call(self,x):
    

    for l in self.mlp:
      x = l(x)

    return x

 
# Fuses together two modalities of data
class Fusion(tfk.Model):
  def __init__(self,
               low_dim):
    super(Fusion,self).__init__()

    self.l = tfkl.Dense(low_dim,kernel_initializer='he_normal')
    self.b = tfkl.BatchNormalization()
    self.r = tfkl.ReLU()
 
  def call(self,inputs):

    return self.l(self.b(self.r(inputs)))


# compresses the modalities of data down to a low dimensional latent space using an mlp style encoder
class Encoder(tfk.Model):
  def __init__(self,
               low_dim,
               nn,
               out_do,
               hidden_do):
    super(Encoder,self).__init__()
    

    self.mlp = []
    for i in range(0,len(nn)):
      self.mlp.append(tfkl.Dense(nn[i],kernel_initializer='he_normal'))
      self.mlp.append(tfkl.BatchNormalization())
      self.mlp.append(tfkl.ReLU())
      self.mlp.append(tfkl.Dropout(out_do*(1-min(1,i)) + hidden_do*(min(1,i))))
  
    self.mlp.append(tfkl.Dense(low_dim,kernel_initializer='he_normal'))
    self.mlp.append(tfkl.BatchNormalization())
    self.mlp.append(tfkl.ReLU())

  def call(self,x):
    

    for l in self.mlp:
      x = l(x)

    return x


# Defines the SynergyIF model
class Fuse(tfk.Model):
  def __init__(self,
               drug_feat,
               cl_feat,
               shape,
               rank,
               X,
               aux_weight,
               low_dim,
               tensor_decoder_nn,
               tensor_decoder_hidden_do,
               tensor_decoder_out_do,
               aux_decoder_hidden_do,
               aux_decoder_out_do,
               encoder_nn,
               encoder_out_do,
               encoder_hidden_do):

    super(Fuse,self).__init__()
    
    self.drug_feat = tf.constant(drug_feat)
    self.cl_feat = tf.constant(cl_feat)
    
    self.aux_weight = aux_weight    

    self.scp = SCP(X=X,shape=shape,rank=rank) 
    
    self.drug_struct_encoder = Encoder(low_dim = low_dim,nn = encoder_nn,out_do = encoder_out_do,hidden_do = encoder_hidden_do)
    self.cell_struct_encoder = Encoder(low_dim = low_dim,nn = encoder_nn,out_do = encoder_out_do,hidden_do = encoder_hidden_do)
    
    self.drug_aux_encoder = Encoder(low_dim = low_dim,nn = encoder_nn,out_do = encoder_out_do,hidden_do = encoder_hidden_do)
    self.cell_aux_encoder = Encoder(low_dim = low_dim,nn = encoder_nn,out_do = encoder_out_do,hidden_do = encoder_hidden_do)
    
    self.drug_fusion = Fusion(low_dim = low_dim)
    self.cell_fusion = Fusion(low_dim = low_dim)

    self.tensor_decoder = TensorDecoder(nn = tensor_decoder_nn,hidden_do = tensor_decoder_hidden_do,out_do = tensor_decoder_out_do)
    self.drug_decoder = AuxDecoder(out_dim = self.drug_feat.shape[-1],nn = encoder_nn[::-1],hidden_do = aux_decoder_hidden_do,out_do = aux_decoder_out_do)
    self.cell_decoder = AuxDecoder(out_dim = self.cl_feat.shape[-1],nn = encoder_nn[::-1],hidden_do = aux_decoder_hidden_do,out_do = aux_decoder_out_do)

  def call(self,indices):
    da_struct_high_dim, db_struct_high_dim, cl_struct_high_dim = self.scp(tf.transpose(indices))
    
    da_aux_high_dim = tf.gather(self.drug_feat,indices[...,0])
    db_aux_high_dim = tf.gather(self.drug_feat,indices[...,1])
    cl_aux_high_dim = tf.gather(self.cl_feat,indices[...,2])

    da_struct_low_dim = self.drug_struct_encoder(da_struct_high_dim) 
    db_struct_low_dim = self.drug_struct_encoder(db_struct_high_dim)
    cl_struct_low_dim = self.cell_struct_encoder(cl_struct_high_dim)

    da_aux_low_dim = self.drug_aux_encoder(da_aux_high_dim)
    db_aux_low_dim = self.drug_aux_encoder(db_aux_high_dim)
    cl_aux_low_dim = self.cell_aux_encoder(cl_aux_high_dim)

    da_fuse = self.drug_fusion(tf.concat([da_aux_low_dim,da_struct_low_dim],axis = -1))
    db_fuse = self.drug_fusion(tf.concat([db_aux_low_dim,db_struct_low_dim],axis = -1))
    cl_fuse = self.cell_fusion(tf.concat([cl_aux_low_dim,cl_struct_low_dim],axis = -1))

    da_recon = self.drug_decoder(da_fuse)
    db_recon = self.drug_decoder(db_fuse)
    cl_recon = self.cell_decoder(cl_fuse)
    tensor_recon = self.tensor_decoder(tf.concat([da_fuse,db_fuse,cl_fuse],axis = -1)) 

    da_recon_loss = 0.5 * tf.reduce_mean((da_recon - da_aux_high_dim)**2)
    db_recon_loss = 0.5 * tf.reduce_mean((db_recon - db_aux_high_dim)**2) 
    cl_recon_loss = 0.5 * tf.reduce_mean((cl_recon - cl_aux_high_dim)**2)

    aux_loss = self.aux_weight * (da_recon_loss + db_recon_loss + cl_recon_loss)
    self.add_loss(aux_loss)
    self.add_metric(aux_loss,name = 'aux_reg')

    return tensor_recon


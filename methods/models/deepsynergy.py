import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.keras.regularizers import l1_l2

# This is the DeepSynergy model
class DeepSynergy(tfk.Model):
  def __init__(self,
               drug_feat=None,
               cl_feat=None,
               shape = None,
               nn = None,
               out_do = None,
               hidden_do = None,
               l1_strength = None,
               l2_strength = None):
    
    super(DeepSynergy,self).__init__()

    self.drug_feat = tf.constant(drug_feat)
    self.cl_feat = tf.constant(cl_feat)
    
    self.mlp = []
    for i in range(0,len(nn)):
      self.mlp.append(tfkl.Dense(nn[i],kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength)))
      if i != (len(nn) -1):  
        self.mlp.append(tfkl.BatchNormalization())
      self.mlp.append(tfkl.ReLU())
      self.mlp.append(tfkl.Dropout(out_do*(1-min(1,i)) + hidden_do*(min(1,i))))
    
    ####### add activation = 'sigmoid'
    self.mlp.append(tfkl.Dense(1,activation='sigmoid', kernel_initializer='he_normal'))

  def call(self,indices):
    
    da = tf.gather(self.drug_feat,indices[...,0])
    db = tf.gather(self.drug_feat,indices[...,1])
    cl = tf.gather(self.cl_feat,indices[...,2])

    x = tf.concat([da,db,cl],axis = -1)
  
    for l in self.mlp:
      x = l(x)

    return x



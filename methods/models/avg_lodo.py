import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import numpy as np
import pandas as pd

# This is the Average model for the Leave Drug Out cross validation strategy
class AvgLODO(tfk.Model):
  def __init__(self,shape,train_set,train_label):
    super(AvgLODO,self).__init__()

    train_set = pd.DataFrame(data=train_set)
    train_set['y'] = train_label
 
    self.avg = np.zeros((shape[0],shape[-1]))
    self.available = np.zeros((shape[0],shape[-1]))
    
    for row in train_set.groupby([0,2]).mean().iterrows():
      self.avg[row[0][0],row[0][1]] = row[1]['y']
    
    for row in train_set.groupby([1,2]).mean().iterrows():
      if self.avg[row[0][0],row[0][1]] != 0:
        self.avg[row[0][0],row[0][1]] += row[1]['y']
        self.avg[row[0][0],row[0][1]] /= 2
      else:
        self.avg[row[0][0],row[0][1]] += row[1]['y']
  
  def call(self,indices):
    
    da = tf.gather_nd(self.avg,tf.stack([indices[...,0],indices[...,-1]],axis = -1))
    db = tf.gather_nd(self.avg,indices[...,1:])
 
    da_available = da != 0
    db_available = db != 0
 
    x = da*tf.cast(da_available,'float64') + db*tf.cast(db_available,'float64')

    return x



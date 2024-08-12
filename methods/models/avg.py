import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import numpy as np
import pandas as pd

# This is the Average model for the Leave Cell Line Out cross validaiton strategy
class Avg(tfk.Model):
  def __init__(self,shape,train_set,train_label):
    super(Avg,self).__init__()

    train_set = pd.DataFrame(data=train_set)
    train_set['y'] = train_label

    self.avg = np.zeros((shape[0],shape[0]))
    
    train_set = train_set.groupby([0,1]).mean()
    for row in train_set.iterrows():
      self.avg[row[0][0],row[0][1]] = row[1]['y']
      self.avg[row[0][1],row[0][0]] = row[1]['y']

  def call(self,indices):
    
    x = tf.gather_nd(self.avg,indices[...,:-1])

    return x



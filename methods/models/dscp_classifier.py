import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.keras.regularizers import l1_l2

def get_dnn(rank,nn,hidden_do,out_do,l1_strength, l2_strength,activation_function):
  
  input_shape = [rank,rank,rank]
  

  inputs = []
  for i in range(3):
      inputs.append(tfkl.Input(shape = [input_shape[i]]))
    
  
  x = tf.concat(inputs,axis = -1)

  # Reshape for MultiHeadAttention
  x = tfkl.Reshape((rank, -1))(x)  

  # MultiHeadAttention layer
  attention_layer = tfkl.MultiHeadAttention(num_heads=2, key_dim=rank) #specify the layer
  x = attention_layer(x, x)

  # Flatten the output
  x = tfkl.Flatten()(x)
  
  for i in range(0,len(nn)):
    x = tfkl.Dense(nn[i],activation=activation_function,kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength))(x)
    if i != (len(nn) -1):  
      x = tfkl.BatchNormalization()(x)
    x = tfkl.ReLU()(x)
    x = tfkl.Dropout(out_do*(1-min(1,i)) + hidden_do*(min(1,i)))(x)
  
  x = tfkl.Dense(1,activation='sigmoid',kernel_initializer='he_normal')(x)
  
  model = tfk.Model(inputs = inputs,outputs = x)

  return model



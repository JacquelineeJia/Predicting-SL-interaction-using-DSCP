import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.keras import regularizers

# This  is the Coupled Matrix Tensor Factorization Model. Withou auxillary data, it becomes the cp-wopt model
class CMTF(tfk.Model):
    def __init__(self,
                 rank,
                 shape,
                 reg = 0,
                 drug_feat=None,
                 cl_feat=None,
                 aux_weight = 0):
        '''
        rank is an important hyperparamater for tensor factorization 
        a higher rank means the model is more powerful but more likely to overfit
        shape is just the shape of the input tensor
        '''

        super(CMTF, self).__init__()

        # Add a dense layer for binary classification with sigmoid activation
        self.classification_layer = tfkl.Dense(1, activation='sigmoid', 
                                               name='classification_layer')
        ####
        self.aux = not drug_feat is None
        self.C_1 = tfkl.Embedding(input_dim = shape[0], 
                                  output_dim=rank,
                                  embeddings_initializer='normal',
                                  trainable = True,
                                  activity_regularizer = regularizers.l2(reg) )
        self.C_2 = self.C_1 # There is a symmetry in the drug drug synergy tensor
        self.C_3 = tfkl.Embedding(input_dim = shape[2],
                                  output_dim=rank,
                                  embeddings_initializer='normal',
                                  trainable = True,
                                  activity_regularizer = regularizers.l2(reg) )
        
        if self.aux:
          self.F_1 = tfkl.Embedding(input_dim = drug_feat.shape[-1], 
                                  output_dim=rank,
                                  embeddings_initializer='normal',
                                  trainable = True,
                                  activity_regularizer = regularizers.l2(reg) )
          self.F_2 = self.F_1
          self.F_3 = tfkl.Embedding(input_dim = cl_feat.shape[-1], 
                                  output_dim=rank,
                                  embeddings_initializer='normal',
                                  trainable = True,
                                  activity_regularizer = regularizers.l2(reg) )
         
        self.C_1.build(None)
        self.C_3.build(None)
 
        if self.aux:
          self.F_1.build(None)
          self.F_3.build(None)

        if self.aux:
          self.drug_feat = tf.constant(drug_feat,dtype = 'float32')
          self.cl_feat = tf.constant(cl_feat,dtype = 'float32')
          self.aux_weight = aux_weight

    def call(self,indices):
        '''
        indices is a three-tuple of indices into the tensor
        such as (1,2,3)

        We will return a number between 0-5 representing the prediction for that index
        of the tensor
        Ex. if indices is (1,2,3) this function will return the predicted review rating that the 1st device
        left on the 2nd website for the 3rd item
        '''

        # select the appropiate rows of the factor matrices
        # c_1,c_2 and c_3 will be sets of vectors with the length of rank
        c_1 = self.C_1(indices[...,0])
        c_2 = self.C_2(indices[...,1])
        c_3 = self.C_3(indices[...,2])

        if self.aux:
          f_1 = self.F_1.embeddings
          f_2 = self.F_2.embeddings
          f_3 = self.F_3.embeddings

          pred_da_feat = tf.einsum('ir,jr->ij',self.C_1.embeddings,f_1)
          pred_db_feat = tf.einsum('ir,jr->ij',self.C_2.embeddings,f_2)
          pred_cl_feat = tf.einsum('ir,jr->ij',self.C_3.embeddings,f_3)

          da_feat_loss = 0.5 * tf.reduce_mean((pred_da_feat - self.drug_feat)**2)
          db_feat_loss = 0.5 * tf.reduce_mean((pred_db_feat - self.drug_feat)**2)
          cl_feat_loss = 0.5 * tf.reduce_mean((pred_cl_feat - self.cl_feat)**2)

          self.add_loss((da_feat_loss + db_feat_loss + cl_feat_loss)*self.aux_weight)
          self.add_metric((da_feat_loss + db_feat_loss + cl_feat_loss),name = 'aux_reg')        

        out = tf.math.reduce_sum(c_1*c_2*c_3,axis = -1)
        ### # Pass 'out' through the classification layer
        prob = self.classification_layer(tf.expand_dims(out, -1))  # Ensure 'out' is the correct shape
        return prob

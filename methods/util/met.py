import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl

def tf_r2(y_true, y_pred):
        SS_res =  tf.reduce_sum(tf.math.square( y_true-y_pred )) 
        SS_tot = tf.reduce_sum(tf.math.square( y_true - tf.reduce_mean(y_true) ) ) 
        return ( 1 - SS_res/(SS_tot + tfk.backend.epsilon()) )

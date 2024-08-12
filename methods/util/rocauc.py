import tensorflow as tf
from keras import backend as K
def calculate_BA(y_true, y_pred):
    # Create an AUC metric instance
    # Create a Precision metric instance
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    y_true = tf.clip_by_value(y_true, 0, 1)

    threshold = 0.5
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    
    cm = tf.math.confusion_matrix(y_true, y_pred_binary, num_classes=2, dtype=tf.float32)
    tn, fp, fn, tp = tf.unstack(tf.reshape(cm, [-1]), num=4)

    sensitivity = tp / (tp + fn+tf.keras.backend.epsilon())
    specificity = tn / (tn + fp + tf.keras.backend.epsilon())

    BA = (sensitivity + specificity)/2
    return BA

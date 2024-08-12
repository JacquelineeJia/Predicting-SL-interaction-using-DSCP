import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.keras.regularizers import l1_l2

def get_cross_attention_dnn(rank, nn, hidden_do, out_do, l1_strength, l2_strength, activation_function):
    # Define input layers for three separate feature matrices
    input_A = tfkl.Input(shape=(rank,))
    input_B = tfkl.Input(shape=(rank,))
    input_C = tfkl.Input(shape= (rank,))

    # Define MultiHeadAttention layer
    attention_layer = tfkl.MultiHeadAttention(num_heads=2, key_dim=rank)
    attention_output_AB = attention_layer(input_A, input_B)  # A attends to B
    attention_output_AC = attention_layer(input_A, input_C)  # A attends to C

    # Ensure the inputs are rank 3 (batch_size, sequence_length, feature_dim)
    input_A = tfkl.Reshape((1, rank))(input_A)
    input_B = tfkl.Reshape((1, rank))(input_B)
    input_C = tfkl.Reshape((1, rank))(input_C)

    # Concatenate attention outputs with original input
    x = tfkl.Concatenate(axis=-1)([input_A, attention_output_AB, attention_output_AC])
    x = tfkl.Flatten()(x)

    # Check if sequence length is variable and adjust accordingly
    if x.shape[1] is None:  # Assuming None represents variable sequence lengths
        x = tfkl.GlobalAveragePooling1D()(x)
    else:
        x = tfkl.Flatten()(x)  # Use flatten only if the sequence length is fixed

    # Passing through Dense layers
    for num_units in nn:
        x = tfkl.Dense(num_units, activation=activation_function, kernel_initializer='he_normal', 
                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_strength, l2=l2_strength))(x)
        x = tfkl.BatchNormalization()(x)
        x = tfkl.ReLU()(x)
        x = tfkl.Dropout(out_do*(1-min(1,num_units)) + hidden_do*(min(1,num_units)))(x)

    x = tfkl.Dense(1, activation='sigmoid')(x)
   
   # Create model
    return tf.keras.Model(inputs=[input_A, input_B, input_C], outputs=x)

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

def mlp_functional(input_shape, hidden_sizes=(32,), activation='tanh', output_activation=None, use_bias=True):
    inputs = tf.keras.Input(input_shape)
    layer=inputs
    for hidden_size in hidden_sizes[:-1]:
        layer = tf.keras.layers.Dense(units=hidden_size, activation=activation)(layer)
    outputs = tf.keras.layers.Dense(
        units=hidden_sizes[-1],
        activation=output_activation,
        use_bias=use_bias,
        # activity_regularizer=regularizers.l2(1e-4)
    )(layer)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def mlp(input_shape, hidden_sizes=(32,), activation='tanh', output_activation=None):

    model = tf.keras.Sequential()
    for h in hidden_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation=activation))
    model.add(tf.keras.layers.Dense(units=hidden_sizes[-1], activation=output_activation))
    model.build(input_shape=(None,) + input_shape)
    return model

"""
Actor-Critics
"""
def mlp_actor_critic(obs_dim, act_dim, hidden_sizes=(64,64), activation='tanh', 
                     output_activation='tanh'):
    with tf.name_scope('pi'):
        pi_network = mlp_functional((obs_dim,), list((32,32))+[act_dim], activation, output_activation, use_bias=False)
    with tf.name_scope('q'):
        q_network = mlp_functional((obs_dim+act_dim,), list(hidden_sizes)+[1], activation, None)
    return pi_network, q_network
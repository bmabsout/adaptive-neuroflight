import numpy as np
import tensorflow as tf


def mlp(input_shape, hidden_sizes=(32,), activation='relu', output_activation=None, clip=False):
    model = tf.keras.Sequential()
    for h in hidden_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation=activation))
    model.add(tf.keras.layers.Dense(units=hidden_sizes[-1], activation=output_activation))
    if clip:
        model.add(tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1.0, 1.0)))
    model.build(input_shape=(None,) + input_shape)
    return model

"""
Actor-Critics
"""
def mlp_actor_critic(obs_dim, act_dim, act_hidden_sizes=(32,32), hidden_sizes=(400,300), activation='relu', 
                     output_activation="tanh"):
    with tf.name_scope('pi'):
        pi_network = mlp((obs_dim,), list(act_hidden_sizes)+[act_dim], activation, output_activation, clip=True)
    with tf.name_scope('q1'):
        q1_network = mlp((obs_dim+act_dim,), list(hidden_sizes)+[1], activation, None)
    with tf.name_scope('q2'):
        q2_network = mlp((obs_dim+act_dim,), list(hidden_sizes)+[1], activation, None)
    return pi_network, q1_network, q2_network
import numpy as np
import tensorflow as tf

def mlp_functional(input_shape, hidden_sizes=(32,), activation='tanh', output_activation=None, use_bias=True, clip=False):
    inputs = tf.keras.Input(input_shape)
    layer=inputs
    for hidden_size in hidden_sizes[:-1]:
        layer = tf.keras.layers.Dense(units=hidden_size, activation=activation)(layer)
    outputs = tf.keras.layers.Dense(
        units=hidden_sizes[-1],
        activation=output_activation,
        use_bias=use_bias,
    )(layer)
    if clip:
        outputs = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1.0, 1.0))(outputs)

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
def mlp_actor_critic(obs_dim, act_dim, actor_hidden_sizes=(64,64), critic_hidden_sizes=(256,256), activation='relu', 
                     output_activation=None):
    # with tf.name_scope('pi'):
    pi_network = mlp_functional((obs_dim,), list(actor_hidden_sizes)+[act_dim], activation, output_activation, use_bias=True, clip=True)
    # with tf.name_scope('q'):
    q_network = mlp_functional((obs_dim+act_dim,), list(critic_hidden_sizes)+[1], activation, "relu")
    return pi_network, q_network
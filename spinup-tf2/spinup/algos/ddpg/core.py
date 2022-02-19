import numpy as np
import tensorflow as tf
import gym

def mlp_functional(inputs, hidden_sizes=(32,), activation='tanh', use_bias=True, output_activation="sigmoid"):
    layer = inputs
    for hidden_size in hidden_sizes[:-1]:
        layer = tf.keras.layers.Dense(units=hidden_size, activation=activation)(layer)
    outputs = tf.keras.layers.Dense(
        units=hidden_sizes[-1],
        activation=output_activation,
        use_bias=use_bias,
    )(layer)

    return outputs

def scale_by_space(scale_me, space): #scale_me: [0,1.0]
    return scale_me*(space.high - space.low) + space.low

def unscale_by_space(unscale_me, space): #outputs [-0.5, 0.5]
    return (unscale_me - space.low)/(space.high - space.low) -0.5
"""
Actor-Critics
"""
def actor(obs_space, act_space, hidden_sizes):
    inputs = tf.keras.Input((obs_space.shape[0],))
    unscaled = unscale_by_space(inputs, obs_space)
    normed = mlp_functional(unscaled, hidden_sizes +(act_space.shape[0],),use_bias=False)
    scaled = scale_by_space(normed, act_space)
    return tf.keras.Model(inputs,scaled)

def critic(obs_space, act_space, hidden_sizes):
    inputs = tf.keras.Input((obs_space.shape[0]+act_space.shape[0],))
    outputs = mlp_functional(inputs, hidden_sizes + (1,), output_activation=None)
    biased_normed = tf.keras.layers.Activation("sigmoid")(outputs - 1.0)
    return tf.keras.Model(inputs, biased_normed)

def mlp_actor_critic(obs_space, act_space, actor_hidden_sizes=(64,64), critic_hidden_sizes=(256,256)):
    return actor(obs_space, act_space, actor_hidden_sizes), critic(obs_space, act_space, critic_hidden_sizes)
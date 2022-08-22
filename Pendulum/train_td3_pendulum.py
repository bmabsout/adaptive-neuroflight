import spinup.algos.td3.td3 as rl_alg
import Pendulum
import numpy as np
import time
import pickle
import tensorflow as tf

def on_save(actor, q_network, epoch, replay_buffer):
    actor.save("pendulum/actor")
    q_network.save("pendulum/critic")
    with open( "pendulum/replay.p", "wb" ) as replay_file:
            pickle.dump( replay_buffer, replay_file)

def existing_actor_critic(*args, **kwargs):
    return tf.keras.models.load_model("right_leaning_pendulum/actor"), tf.keras.models.load_model("right_leaning_pendulum/critic")

rl_alg.td3(lambda: Pendulum.PendulumEnv(g=10.0, setpoint=0.0), seed=0)
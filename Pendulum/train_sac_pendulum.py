import spinup.algos.sac.sac as rl_alg
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

rl_alg.sac(lambda: Pendulum.PendulumEnv(g=10.0, setpoint=np.pi/4.0), seed=0, 
        steps_per_epoch=1000, epochs=200, replay_size=int(1e5), gamma=0.9, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=200, start_steps=1000, 
        update_after=900, update_every=50, num_test_episodes=4, num_opt_steps=30, max_ep_len=200, save_freq=1)
import spinup.algos.ddpg.ddpg as rl_alg
import Pendulum
import numpy as np
import tensorflow as tf

def on_save(actor, q_network, epoch):
    actor.save("pend_actor")
    q_network.save("q_val")

def existing_actor_critic(*args, **kwargs):
    return tf.keras.models.load_model("right_leaning_actor"), tf.keras.models.load_model("right_leaning_q")

rl_alg.ddpg(lambda: Pendulum.PendulumEnv(g=10.0, setpoint=+np.pi/4.0), hp=rl_alg.HyperParams(gamma=0.9, max_ep_len=200,epochs=10, q_lr=1e-3, start_steps=1000), on_save=on_save, actor_critic=existing_actor_critic, safety_q=tf.keras.models.load_model("right_leaning_q")) 
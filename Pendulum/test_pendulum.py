import tensorflow as tf
import numpy as np
import Pendulum

saved = tf.saved_model.load("pend_actor")
actor = lambda x: saved(np.array([x]))[0]
env = Pendulum.PendulumEnv(g=10., color=(0.0, 0.8,0.2))
o = env.reset()
for _ in range(200):
    o, r, d, i, = env.step(actor(o))
    env.render()


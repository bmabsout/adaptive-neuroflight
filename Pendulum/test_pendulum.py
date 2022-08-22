import tensorflow as tf
import numpy as np
import Pendulum
import matplotlib.pyplot as plt
import pickle
import time

# saved = tf.saved_model.load("right_leaning_pendulum/actor")
saved = tf.keras.models.load_model("sac_PendulumEnv_seed=0_steps_per_epoch=1000_epochs=100_gamma=0.9_lr=0.001_batch_size=200_start_steps=1000_update_after=900_update_every=50")
# saved = tf.saved_model.load("pretty_please")

actor = lambda x: saved(np.array([x]))[0]
env = Pendulum.PendulumEnv(g=10., color=(0.0, 0.8,0.2))
env.seed(123)
o = env.reset()
env.render()
time.sleep(0.5)
high = env.action_space.high
low = env.action_space.low
os = []
for _ in range(200):
    o, r, d, i, = env.step(actor(o)*(high - low)/2.0 + (high + low)/2.0)
    os.append(o)
    env.render()


# with open("up_leaning.p", "wb") as file:
# 	pickle.dump(np.array(os), file)

# plt.plot(np.array(os)[:,1])
# plt.show()
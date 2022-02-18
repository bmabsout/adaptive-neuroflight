import gym
import numpy as np
import tensorflow as tf
import neuroflight_trainer.gyms


env = gym.make("gymfc_perlin_discontinuous-v3")
# env = gym.make("LunarLanderContinuous-v2")
# pretty_please = tf.keras.models.load_model("LL_pretty_please")
pretty_please = tf.keras.models.load_model("pretty_please")
# env.seed(1234)
o = env.reset()
for i in range(20000):
	a = pretty_please.predict(np.array([o]))[0]
	o, r, d, info = env.step(a)
	#env.render()
	print(o, r, info)
	print("action:", a)

import spinup
import numpy as np
import tensorflow as tf
import gym
import time
import neuroflight_trainer.gyms
import os

# from spinup.utils.run_utils import setup_logger_kwargs
seed = 1234

spinup.ddpg(lambda : gym.make("gymfc_perlin_discontinuous-v3"), actor_critic=spinup.algos.ddpg.core.mlp_actor_critic,
    ac_kwargs=dict(hidden_sizes=[256,256]), gamma=0.9, pi_lr=1e-4, q_lr=1e-4,
    seed=seed, steps_per_epoch=20000, epochs=25, max_ep_len=10000)
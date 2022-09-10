import numpy as np
import gym
import shutil
import time
import os
from signal import signal, SIGINT

import spinup
import neuroflight_trainer.gyms
import neuroflight_trainer.variables as variables
from neuroflight_trainer.rl_algs import training_utils
from neuroflight_trainer.validation.fc_logging_utils import FlightLog

import tensorflow as tf

def save_flight(env, seed, actor, save_location, num_episodes=3):
    print("saving to:", save_location)
    actor.save(save_location + "/actor")
    flight_log = FlightLog(save_location)
    num_total_steps = 0
    rewards_sum = 0
    for episode_index in range(num_episodes):
        env.ep_counter = 0
        env.seed(seed+episode_index)
        ob = env.reset()
        done = False
        while not done:
            ac = actor(np.array([ob]))[0].numpy()
            ob, composed_reward, done, ep_info = env.step(ac)
            num_total_steps += 1
            rewards_sum += composed_reward
            flight_log.add(ob, 
                           composed_reward, 
                           ep_info['rewards_dict'],
                           #ac,
                           env.y,
                           env.imu_angular_velocity_rpy, 
                           env.omega_target,
                           env.esc_motor_angular_velocity,
                           dbg=env.dbg)
        flight_log.save(episode_index, ob.shape[0])
    return rewards_sum/num_total_steps

def train_nf1(seed=int(time.time()* 1e5) % int(1e6), **kwargs):
    signal(SIGINT, training_utils.handler)

    
    training_dir = training_utils.get_training_dir('tf2_sac', seed)
    print ("Storing results to ", training_dir)


    ckpt_dir = os.path.join(training_dir, "checkpoints")
    os.makedirs(training_dir)


    env_id = "gymfc_perlin_discontinuous-v3"
    env = gym.make(env_id)
    env.seed(seed)
    env.noise_sigma = 1

    test_env = gym.make(env_id)
    test_env.noise_sigma = 1
    def on_save(actor, critic, ckpt_id, rb):
        return save_flight(test_env, seed, actor, os.path.join(ckpt_dir, f"ckpt_{ckpt_id}"))

    def on_save_sac(ac, ckpt_id, rb):
        return save_flight(test_env, seed, ac.pi.deterministic_actor, os.path.join(ckpt_dir, f"ckpt_{ckpt_id}"))

    def on_save_ppo(actor, ckpt_id):
        return save_flight(test_env, seed, actor, os.path.join(ckpt_dir, f"ckpt_{ckpt_id}"))

    final_actor = spinup.sac(
        lambda : env,
        # hp=spinup.algos.ddpg.ddpg.HyperParams(ac_kwargs={
        #     "actor_hidden_sizes":(32,32),
        #     "critic_hidden_sizes":(400,300),
        #     "obs_normalizer": np.array([500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0])
        # },**kwargs),
        on_save=on_save_sac,
        **kwargs
    )

    return save_flight(test_env, seed, final_actor, os.path.join(ckpt_dir, f"ckpt_{kwargs['epochs']}"))

if __name__ == '__main__':
    hypers = {
        'steps_per_epoch': 10000,
        'replay_size': 5000000,
        'gamma': 0.95,
        'polyak': 0.995,
        'pi_lr': 0.001, #tf.optimizers.schedules.PolynomialDecay(3e-4, 1e6, end_learning_rate=0),
        'q_lr': 0.001, #tf.optimizers.schedules.PolynomialDecay(3e-4, 1e6, end_learning_rate=0),
        'batch_size': 200,
        'act_noise': 0.05,
        'max_ep_len': 10000,
        'epochs': 100
    }
    
    train_nf1(**hypers)

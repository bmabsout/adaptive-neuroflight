# import spinup
# import numpy as np
# import tensorflow as tf
# import gym
# import time
# import neuroflight_trainer.gyms
# import os

# # from spinup.utils.run_utils import setup_logger_kwargs
# seed = 1234

# spinup.ddpg(
#     lambda : gym.make("gymfc_perlin_discontinuous-v3"),
#     actor_critic=spinup.algos.ddpg.core.mlp_actor_critic,
#     gamma=0.9,
#     pi_lr=5e-4,#tf.optimizers.schedules.PolynomialDecay(3e-4, 1e6, end_learning_rate=0),
#     q_lr=1e-3,#tf.optimizers.schedules.PolynomialDecay(3e-4, 1e6, end_learning_rate=0),
#     seed=seed,
#     steps_per_epoch=10000,
#     epochs=100,
#     max_ep_len=10000
# )

import numpy as np
import gym
import shutil
import time
import os
from signal import signal, SIGINT

import spinup
from spinup.algos.ddpg.ddpg import HyperParams
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
    
def existing_actor_critic(*args, **kwargs):
    actor = tf.keras.models.load_model(
        "/data/neuroflight/CODE/gymfc-nf1/training_data/results/tf2_ddpg_c9188dd6_s83352_t220224-143501/checkpoints/ckpt_7/actor"
    )
    critic = tf.keras.models.load_model(
        "/data/neuroflight/CODE/gymfc-nf1/training_data/results/tf2_ddpg_c9188dd6_s83352_t220224-143501/checkpoints/ckpt_7/critic"
    )
    return actor, critic 

def train_nf1(hypers):
    signal(SIGINT, training_utils.handler)

    
    training_dir = training_utils.get_training_dir('tf2_ddpg', hypers.seed)
    print ("Storing results to ", training_dir)


    ckpt_dir = os.path.join(training_dir, "checkpoints")
    os.makedirs(training_dir)


    env_id = "gymfc_perlin_discontinuous-v3"
    env = gym.make(env_id)
    env.seed(hypers.seed)


    env.noise_sigma = 1

    test_env = gym.make(env_id)
    test_env.noise_sigma = 1
    avg_reward = 0
    def on_save(actor, critic, ckpt_id):
        save_path = os.path.join(ckpt_dir, f"ckpt_{ckpt_id}")
        critic.save(os.path.join(save_path, "critic"))
        avg_reward = save_flight(test_env, hypers.seed, actor, save_path)

    spinup.ddpg(
        lambda : env,
        on_save=on_save,
        # actor_critic=existing_actor_critic,
        hp=hypers
    )
    return avg_reward

if __name__ == '__main__':
    # tf.debugging.experimental.enable_dump_debug_info(
    #     "/tmp/tfdbg4_logdir",
    #     tensor_debug_mode="FULL_HEALTH",
    #     circular_buffer_size=-1)
    hypers = HyperParams(
        steps_per_epoch=10000,
        start_steps=10000,
        replay_size=1000000,
        gamma=0.9,
        polyak=0.995,
        pi_lr=0.001, #tf.optimizers.schedules.PolynomialDecay(3e-4, 1e6, end_learning_rate=0),
        q_lr=0.001, #tf.optimizers.schedules.PolynomialDecay(3e-4, 1e6, end_learning_rate=0),
        batch_size=200,
        act_noise=0.1,
        max_ep_len=10000,
        epochs=100
    )
    
    train_nf1(hypers)

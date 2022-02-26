from dataclasses import asdict, dataclass
from typing import NamedTuple
import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.ddpg import core
from spinup.utils.logx import EpochLogger

def with_importance(x, importance):
    return (x-importance)/(1.0-importance)

def geo(l, axis):
    return np.exp(np.mean(np.log(l), axis=axis))

def p_mean(l, p, slack=0.0):
    slacked = np.array(l) + slack
    if(len(slacked.shape) == 1): #enforce having batches
        slacked = np.array([slacked])
    batch_size = slacked.shape[0]
    res = np.zeros(batch_size)
    handle_zeros = (slacked > 1e-20).all(axis=1) if p <=1e-20 else np.full(batch_size, True)

    res[handle_zeros] = (geo(slacked[handle_zeros], axis=1) if p == 0 else np.mean(slacked[handle_zeros]**p, axis=1)**(1.0/p)) - slack
    return res.squeeze()

def to_positive(r):
    return np.clip(1-r,0,1)

def closeness_rw(true_error):
    return p_mean(to_positive(np.tanh(np.abs(true_error)/100)),0)

def acc_rw(motor_acc):
    return with_importance(p_mean(to_positive(np.abs(motor_acc)),0),-0.7)

def rewards_scalar(rewards_list):
    closeness, keep_middle, acc = rewards_list
    return p_mean([closeness,acc], 0)

def unroll_rpy(rpy):
    return np.array([rpy.roll, rpy.pitch, rpy.yaw])

def unroll_act(act):
    return np.array([
        act.top_left,
        act.top_right,
        act.bottom_left,
        act.bottom_right,
    ])

def unroll_obs(obs):
    return np.concatenate([
        unroll_rpy(obs.error),
        unroll_rpy(obs.ang_vel),
        unroll_rpy(obs.ang_acc),
        unroll_act(obs.prev_action)
    ])


def rewards_fn(obs, act):
    motor_acc = unroll_act(obs.prev_action) - unroll_act(act)
    acc = acc_rw(motor_acc)
    closeness = closeness_rw(unroll_rpy(obs.error))
    return p_mean([closeness, acc], 0)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class HyperParams:
    def __init__( self,
            ac_kwargs={"actor_hidden_sizes":(32,32), "critic_hidden_sizes":(400,300)},
            seed=int(time.time()* 1e5) % int(1e6),
            steps_per_epoch=1000,
            replay_size=int(1e6),
            gamma=0.9,
            polyak=0.995,
            pi_lr=1e-4,
            q_lr=1e-4,
            batch_size=200,
            train_every=50,
            train_steps=30,
        ):
        self.ac_kwargs = ac_kwargs
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.batch_size = batch_size
        self.train_every = train_every
        self.train_steps = train_steps
    
"""

Deep Deterministic Policy Gradient (DDPG)

"""
def live_ddpg(obs_queue, obs_space, act_space, hp: HyperParams=HyperParams(),actor_critic=core.mlp_actor_critic, logger_kwargs=dict(), save_freq=4, on_save=lambda *_:(), safety_q=None):
    """

    Args:
        obs_queue : a queue containing (observation1, action, observation2) objects

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """


    logger = EpochLogger(**logger_kwargs)
    logger.save_config(hp.__dict__)

    tf.random.set_seed(hp.seed)
    np.random.seed(hp.seed)

    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    max_q_val = 1.0/(1.0-hp.gamma)

    # Main outputs from computation graph
    with tf.name_scope('main'):
        pi_network, q_network = actor_critic(obs_space, act_space, **hp.ac_kwargs)
    
    # Target networks
    with tf.name_scope('target'):
        # Note that the action placeholder going to actor_critic here is 
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        pi_targ_network, q_targ_network  = actor_critic(obs_space, act_space, **hp.ac_kwargs)

    # make sure network and target network is using the same weights
    pi_targ_network.set_weights(pi_network.get_weights())
    q_targ_network.set_weights(q_targ_network.get_weights())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=hp.replay_size)

    # Separate train ops for pi, q
    pi_optimizer = tf.keras.optimizers.Adam(learning_rate=hp.pi_lr)
    q_optimizer = tf.keras.optimizers.Adam(learning_rate=hp.q_lr)

    # Polyak averaging for target variables
    @tf.function
    def target_update():
        for v_main, v_targ in zip(pi_network.trainable_variables, pi_targ_network.trainable_variables):
            v_targ.assign(hp.polyak*v_targ + (1-hp.polyak)*v_main)
        for v_main, v_targ in zip(q_network.trainable_variables, q_targ_network.trainable_variables):
            v_targ.assign(hp.polyak*v_targ + (1-hp.polyak)*v_main)

    @tf.function
    def q_update(obs1, obs2, acts, rews, dones):
        with tf.GradientTape() as tape:
            q = tf.squeeze(q_network(tf.concat([obs1, acts], axis=-1)), axis=1)
            pi_targ = pi_targ_network(obs2)
            q_pi_targ = tf.squeeze(q_targ_network(tf.concat([obs2, pi_targ], axis=-1)), axis=1)
            backup = tf.stop_gradient(rews/max_q_val + hp.gamma * q_pi_targ)
            q_loss = tf.reduce_mean((q-backup)**2) + sum(q_network.losses)*0.1
        grads = tape.gradient(q_loss, q_network.trainable_variables)
        grads_and_vars = zip(grads, q_network.trainable_variables)
        q_optimizer.apply_gradients(grads_and_vars)
        return q_loss, q

    @tf.function
    def pi_update(obs):
        with tf.GradientTape() as tape:
            pi = pi_network(obs)
            q_pi = tf.reduce_mean(tf.squeeze(q_network(tf.concat([obs, pi], axis=-1)), axis=-1))
            if safety_q:
                safe_q_pi = tf.reduce_mean(tf.squeeze(safety_q(tf.concat([obs, pi], axis=-1)), axis=-1))
                powers = 0.5
            else:
                powers = 1.0
                safe_q_pi = 1.0
            # tf.print("pi", pi[100])
            # tf.print("extremes", 1.0 - extremes[100])
            # extremes = (tf.maximum(tf.abs(pi), 0.9)-0.9)/0.11

            avoid_extremes = tf.reduce_min(1.0 - tf.abs(pi)**2.0)
            # tf.print("avoid_extremes", avoid_extremes)
            # tf.print("q_pi", q_pi)
            reg = sum(pi_network.losses)*0.015
            pi_loss = 1.0 - (q_pi*safe_q_pi)**powers + reg
        grads = tape.gradient(pi_loss, pi_network.trainable_variables)
        grads_and_vars = zip(grads, pi_network.trainable_variables)
        pi_optimizer.apply_gradients(grads_and_vars)
        return pi_loss, avoid_extremes, q_pi, safe_q_pi, reg

    start_time = time.time()
    t = 0
    # Main loop: collect experience in env and update/log each epoch
    while True:
        (o, a, o2) = obs_queue.get()
        t+=1
        r = rewards_fn(o, a)
        d = False
        # Store experience to replay buffer

        replay_buffer.store(unroll_obs(o), unroll_act(a), r, unroll_obs(o2), d)

        if t % hp.train_every == 0:
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for _ in range(hp.train_steps):
                batch = replay_buffer.sample_batch(hp.batch_size)

                obs1 = tf.constant(batch['obs1'])
                obs2 = tf.constant(batch['obs2'])
                acts = tf.constant(batch['acts'])
                rews = tf.constant(batch['rews'])
                dones = tf.constant(batch['done'])
                # Q-learning update
                loss_q, q_vals = q_update(obs1, obs2, acts, rews, dones)
                logger.store(LossQ=loss_q)

                # Policy update
                pi_loss, avoid_extremes, qs, safe_qs, reg = pi_update(obs1)
                logger.store(
                    LossPi=pi_loss.numpy(),
                    NormQ=qs,
                    NormSafe=safe_qs,
                    AvoidExtremes=avoid_extremes,
                    Reg=reg
                )
                # target update
                target_update()

        # End of epoch wrap-up
        if t > 0 and t % hp.steps_per_epoch == 0:
            epoch = t // hp.steps_per_epoch

            # Save model
            if (epoch % save_freq == 0):
                on_save(pi_targ_network, q_network, epoch//save_freq)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('NormQ', average_only=True)
            logger.log_tabular('NormSafe', average_only=True)
            logger.log_tabular('AvoidExtremes', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('Reg', average_only=True)
            logger.dump_tabular()
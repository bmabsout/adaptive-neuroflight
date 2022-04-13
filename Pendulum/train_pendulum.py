import spinup.algos.ddpg.ddpg as rl_alg
import Pendulum
import numpy as np
import tensorflow as tf

def on_save(actor, q_network, epoch):
    actor.save("mid_leaning_actor")
    q_network.save("mid_leaning_q")

def existing_actor_critic(*args, **kwargs):
    return tf.keras.models.load_model("right_leaning_actor"), tf.keras.models.load_model("right_leaning_q")

rl_alg.ddpg(lambda: Pendulum.PendulumEnv(g=10.0, setpoint=0.0)
	, hp = rl_alg.HyperParams(
        steps_per_epoch=1000,
        ac_kwargs={
            "actor_hidden_sizes":(32,32),
            "critic_hidden_sizes":(256,256),
            "obs_normalizer": np.array([1.0, 1.0, 8.0])
        },
        start_steps=1000,
        replay_size=500000,
        gamma=0.9,
        polyak=0.995,
        pi_lr=tf.optimizers.schedules.PolynomialDecay(1e-3, 1e6, end_learning_rate=0),
        q_lr=tf.optimizers.schedules.PolynomialDecay(1e-3, 1e6, end_learning_rate=0),
        batch_size=200,
        act_noise=0.1,
        max_ep_len=200,
        epochs=10
    )
	, on_save=on_save
)
	# , actor_critic=existing_actor_critic
	# , safety_q=tf.keras.models.load_model("right_leaning_q")) 


import tensorflow as tf
import numpy as np


# @tf.function
def geo(l,axis=0):
    return tf.exp(tf.reduce_mean(tf.math.log(l),axis=axis))

# @tf.function
def p_mean(l, p, slack=0.0, axis=1):
    slacked = l + slack
    if(len(slacked.shape) == 1): #enforce having batches
        slacked = tf.expand_dims(slacked, axis=0)
    batch_size = slacked.shape[0]
    zeros = tf.zeros(batch_size, l.dtype)
    handle_zeros = tf.reduce_all(slacked > 1e-20, axis=axis) if p <=1e-20 else tf.fill((batch_size,), True)
    escape_from_nan = tf.where(tf.expand_dims(handle_zeros, axis=axis), slacked, slacked*0.0 + 1.0)
    handled = (
            geo(escape_from_nan, axis=axis)
        if p == 0 else
            tf.reduce_mean(escape_from_nan**p, axis=axis)**(1.0/p)
        ) - slack
    res = tf.where(handle_zeros, handled, zeros)
    return res

# @tf.function
def p_to_min(l, p=0, q=0.2):
    deformator = p_mean(1.0-l, q)
    return p_mean(l, p)*deformator + (1.0-deformator)*tf.reduce_min(l)

# @tf.function
def weaken(weaken_me, weaken_by):
    return (weaken_me + weaken_by)/(1.0 + weaken_by)

@tf.custom_gradient
def to_constraint(x):
  def grad(dy):
    return dy
  return tf.math.sigmoid(x), grad

init_x = tf.Variable(0.001)
init_y = tf.Variable(-10.0)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

for i in range(15):
	with tf.GradientTape() as tape:
		x = tf.math.tanh(init_x)
		y = tf.grad_pass_through(tf.math.sigmoid)(init_y)#to_constraint(init_y)
		gain = p_mean(tf.stack([x,y]), 0.0)

	vars = [init_x, init_y]
	# gradients = list(map(tf.math.tanh, tape.gradient(gain, vars)))
	gradients = tape.gradient(gain, vars)
	optimizer.apply_gradients(zip(map(lambda g: -g, gradients), vars), vars)

	print(list(map(lambda g: g.numpy(), gradients)))
	print("x", x.numpy())
	print("y", y.numpy())
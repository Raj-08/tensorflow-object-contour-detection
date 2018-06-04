import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

DEFAULT_PADDING = 'SAME'

def make_var(name, shape,trainable=True):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=trainable)

def batch_normalization(input, name, is_training, activation_fn=None, scale=True):
        with tf.variable_scope(name) as scope:
            output = slim.batch_norm(
                input,
                activation_fn=activation_fn,
                is_training=is_training,
                updates_collections=None,
                scale=scale,
                scope=scope)
            return output
        
def conv(input,k_h,k_w,c_o,s_h,s_w,name,relu=True,padding=DEFAULT_PADDING,group=1,biased=True):
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = make_var('weights', shape=[k_h, k_w, c_i / group, c_o],trainable=True)
            if group == 1:
                output = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                output = tf.concat(3, output_groups)
            if biased:
                biases = make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

def relu(input):
    return tf.nn.relu(input)

def sigmoid(input, name):
    return tf.nn.sigmoid(input, name=name)    

def unpool_with_argmax(pool, ind, name = None, ksize=[1, 2, 2, 1]):
    with tf.variable_scope(name):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]
        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret
    
def dropout(input, name):
        #keep = 1 - use_dropout + (use_dropout * keep_prob)
        return tf.nn.dropout(input,0.75, name=name)

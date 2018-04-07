import numpy as np
import tensorflow as tf
import lightsaber.tensorflow.util as util


def build_train(model,
                num_actions,
                optimizer,
                lstm_unit=256,
                state_shape=[84, 84, 1],
                grad_clip=40.0,
                value_factor=0.5,
                policy_factor=1.0,
                entropy_factor=0.01,
                scope='a3c',
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # placeholers
        obs_input = tf.placeholder(tf.float32, [None] + state_shape, name='obs')
        rnn_state_ph0 = tf.placeholder(
            tf.float32, [1, lstm_unit], name='rnn_state_0')
        rnn_state_ph1 = tf.placeholder(
            tf.float32, [1, lstm_unit], name='rnn_state_1')
        actions_ph = tf.placeholder(tf.uint8, [None], name='action')
        target_values_ph = tf.placeholder(tf.float32, [None], name='value')
        advantages_ph = tf.placeholder(tf.float32, [None], name='advantage')

        # rnn state in tuple
        rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(
            rnn_state_ph0, rnn_state_ph1)

        # network outpus
        policy, value, state_out = model(
            obs_input, rnn_state_tuple, num_actions, lstm_unit, scope='model')

        actions_one_hot = tf.one_hot(actions_ph, num_actions, dtype=tf.float32)
        log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
        log_prob = tf.reduce_sum(log_policy * actions_one_hot, [1])

        # loss
        value_loss = tf.reduce_sum(
            tf.square(target_values_ph - tf.reshape(value, [-1])))
        entropy = -tf.reduce_sum(policy * log_policy)
        advantages  = tf.reshape(advantages_ph, [-1, 1])
        policy_loss = -tf.reduce_sum(log_prob * advantages)
        loss = value_factor * value_loss\
            + policy_factor * policy_loss\
            - entropy_factor * entropy

        # local network weights
        local_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        # global network weights
        global_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

        # gradients
        gradients, _ = tf.clip_by_global_norm(
            tf.gradients(loss, local_vars), grad_clip)

        optimize_expr = optimizer.apply_gradients(zip(gradients, global_vars))

        update_local_expr = []
        for local_var, global_var in zip(local_vars, global_vars):
            update_local_expr.append(local_var.assign(global_var))
        update_local_expr = tf.group(*update_local_expr)
        update_local = util.function([], [], updates=[update_local_expr])

        train = util.function(
            inputs=[
                obs_input, rnn_state_ph0, rnn_state_ph1,
                actions_ph, target_values_ph, advantages_ph
            ],
            outputs=[loss],
            updates=[optimize_expr]
        )

        action_dist = util.function(
            [obs_input, rnn_state_ph0, rnn_state_ph1], policy)

        act = util.function(
            inputs=[obs_input, rnn_state_ph0, rnn_state_ph1],
            outputs=[policy, value, state_out]
        )

    return act, train, update_local, action_dist

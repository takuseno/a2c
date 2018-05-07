import numpy as np
import tensorflow as tf


def build_train(model,
                num_actions,
                optimizer,
                nenvs,
                lstm_unit=256,
                state_shape=[84, 84, 1],
                grad_clip=40.0,
                value_factor=0.5,
                policy_factor=1.0,
                entropy_factor=0.01,
                scope='a2c',
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # placeholers
        obs_input = tf.placeholder(tf.float32, [None] + state_shape, name='obs')
        rnn_state_ph0 = tf.placeholder(
            tf.float32, [nenvs, lstm_unit], name='rnn_state_0')
        rnn_state_ph1 = tf.placeholder(
            tf.float32, [nenvs, lstm_unit], name='rnn_state_1')
        actions_ph = tf.placeholder(tf.uint8, [None], name='action')
        target_values_ph = tf.placeholder(tf.float32, [None], name='value')
        advantages_ph = tf.placeholder(tf.float32, [None], name='advantage')
        step_size_ph = tf.placeholder(tf.int32, [], name='step_size')
        mask_ph = tf.placeholder(tf.float32, [None], name='mask')

        # rnn state in tuple
        rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(
            rnn_state_ph0, rnn_state_ph1)

        # network outpus
        policy, value, state_out = model(
            obs_input, rnn_state_tuple, num_actions,
            lstm_unit, nenvs, step_size_ph, scope='model')

        actions_one_hot = tf.one_hot(actions_ph, num_actions, dtype=tf.float32)
        log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
        log_prob = tf.reduce_sum(log_policy * actions_one_hot, [1])

        # loss
        advantages = tf.reshape(advantages_ph, [-1, 1])
        target_values = tf.reshape(target_values_ph, [-1, 1])
        masks = tf.reshape(mask_ph, [-1, 1])
        with tf.variable_scope('value_loss'):
            value_loss = tf.reduce_sum(tf.square(target_values - value) * masks)
        with tf.variable_scope('entropy'):
            entropy = -tf.reduce_sum(
                tf.reduce_sum(policy * log_policy, axis=1) * masks)
        with tf.variable_scope('policy_loss'):
            policy_loss = tf.reduce_sum(log_prob * advantages * masks)
        loss = value_factor * value_loss\
            - policy_factor * policy_loss\
            - entropy_factor * entropy

        # network weights
        network_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        # gradients
        gradients, _ = tf.clip_by_global_norm(
            tf.gradients(loss, network_vars), grad_clip)

        optimize_expr = optimizer.apply_gradients(zip(gradients, network_vars))

        def train(obs, actions, targets, advantages, rnn_state0, rnn_state1, masks, step_size):
            feed_dict = {
                obs_input: obs,
                actions_ph: actions,
                target_values_ph: targets,
                advantages_ph: advantages,
                rnn_state_ph0: rnn_state0,
                rnn_state_ph1: rnn_state1,
                mask_ph: masks,
                step_size_ph: step_size
            }
            loss_val, _ = tf.get_default_session().run([loss, optimize_expr], feed_dict=feed_dict)
            return loss_val

        def act(obs, rnn_state0, rnn_state1):
            feed_dict = {
                obs_input: obs,
                rnn_state_ph0: rnn_state0,
                rnn_state_ph1: rnn_state1,
                step_size_ph: 1
            }
            return tf.get_default_session().run(
                [policy, value, state_out], feed_dict=feed_dict)

    return act, train

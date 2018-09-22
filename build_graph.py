import numpy as np
import tensorflow as tf


def build_train(model,
                num_actions,
                optimizer,
                nenvs,
                step_size,
                lstm_unit=256,
                state_shape=[84, 84, 1],
                grad_clip=40.0,
                value_factor=0.5,
                entropy_factor=0.01,
                scope='a2c',
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # placeholers
        step_obs_input = tf.placeholder(
            tf.float32, [nenvs] + state_shape, name='obs')
        train_obs_input = tf.placeholder(
            tf.float32, [nenvs*step_size] + state_shape, name='obs')
        rnn_state_ph = tf.placeholder(
            tf.float32, [nenvs, lstm_unit*2], name='rnn_state')
        actions_ph = tf.placeholder(tf.uint8, [None], name='action')
        returns_ph = tf.placeholder(tf.float32, [None], name='returns')
        advantages_ph = tf.placeholder(tf.float32, [None], name='advantage')
        masks_ph = tf.placeholder(tf.float32, [nenvs * step_size], name='mask')

        # network outpus for inference
        step_policy, step_value, state_out = model(
            step_obs_input, tf.constant(0.0, shape=[nenvs, 1]), rnn_state_ph,
            num_actions, lstm_unit, nenvs, 1, scope='model')
        # network outputs for training
        train_policy, train_value, _ = model(
            train_obs_input, tf.reshape(masks_ph, [nenvs * step_size, 1]),
            rnn_state_ph, num_actions, lstm_unit, nenvs, step_size,
            scope='model')

        actions_one_hot = tf.one_hot(actions_ph, num_actions, dtype=tf.float32)
        log_policy = tf.log(tf.clip_by_value(train_policy, 1e-20, 1.0))
        log_prob = tf.reduce_sum(log_policy * actions_one_hot, axis=1, keep_dims=True)

        # loss
        advantages = tf.reshape(advantages_ph, [-1, 1])
        returns = tf.reshape(returns_ph, [-1, 1])
        with tf.variable_scope('value_loss'):
            value_loss = tf.reduce_mean(tf.square(returns - train_value))
            value_loss *= value_factor
        with tf.variable_scope('entropy'):
            entropy = -tf.reduce_mean(tf.reduce_sum(train_policy * log_policy, axis=1))
            entropy *= entropy_factor
        with tf.variable_scope('policy_loss'):
            policy_loss = tf.reduce_mean(log_prob * advantages)
        loss = value_loss - policy_loss - entropy

        # network weights
        network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        # gradients
        gradients = tf.gradients(loss, network_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
        # update
        grads_and_vars = zip(clipped_gradients, network_vars)
        optimize_expr = optimizer.apply_gradients(grads_and_vars)

        def train(obs, actions, returns, advantages, rnn_state, masks):
            feed_dict = {
                train_obs_input: obs,
                actions_ph: actions,
                returns_ph: returns,
                advantages_ph: advantages,
                rnn_state_ph: rnn_state,
                masks_ph: masks
            }
            sess = tf.get_default_session()
            return sess.run([loss, optimize_expr], feed_dict=feed_dict)[0]

        def act(obs, rnn_state):
            feed_dict = {
                step_obs_input: obs,
                rnn_state_ph: rnn_state
            }
            sess = tf.get_default_session()
            ops = [step_policy, step_value, state_out]
            return sess.run(ops, feed_dict=feed_dict)

    return act, train

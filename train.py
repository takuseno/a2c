import threading
import multiprocessing
import argparse
import cv2
import gym
import copy
import os
import time
import constants
import numpy as np
import tensorflow as tf

from lightsaber.tensorflow.util import initialize
from lightsaber.tensorflow.log import TfBoardLogger, dump_constants
from lightsaber.rl.trainer import BatchTrainer
from lightsaber.rl.env_wrapper import EnvWrapper, BatchEnvWrapper
from actions import get_action_space
from network import make_network
from agent import Agent
from datetime import datetime


def main():
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongDeterministic-v4')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--load', type=str)
    parser.add_argument('--logdir', type=str, default=date)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results/' + args.logdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + args.logdir)

    # save settings
    dump_constants(constants, os.path.join(outdir, 'constants.json'))

    sess = tf.Session()
    sess.__enter__()

    model = make_network(constants.CONVS, lstm=constants.LSTM)

    # share Adam optimizer with all threads!
    lr = tf.Variable(constants.LR)
    decayed_lr = tf.placeholder(tf.float32)
    decay_lr_op = lr.assign(decayed_lr)
    optimizer = tf.train.AdamOptimizer(lr)

    env_name = args.env
    actions = get_action_space(env_name)
    agent = Agent(
        model,
        actions,
        optimizer,
        nenvs=constants.ACTORS,
        gamma=constants.GAMMA,
        lstm_unit=constants.LSTM_UNIT,
        time_horizon=constants.TIME_HORIZON,
        policy_factor=constants.POLICY_FACTOR,
        value_factor=constants.VALUE_FACTOR,
        entropy_factor=constants.ENTROPY_FACTOR,
        grad_clip=constants.GRAD_CLIP,
        state_shape=constants.IMAGE_SHAPE + [constants.STATE_WINDOW]
    )

    saver = tf.train.Saver()
    if args.load:
        saver.restore(sess, args.load)

    def s_preprocess(state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, tuple(constants.IMAGE_SHAPE))
        state = np.array(state, dtype=np.float32)
        return state / 255.0

    # create environemtns
    envs = [EnvWrapper(gym.make(args.env)) for _ in range(constants.ACTORS)]
    batch_env = BatchEnvWrapper(
        envs,
        r_preprocess=lambda r: np.clip(r, -1.0, 1.0),
        s_preprocess=s_preprocess
    )

    initialize()

    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    logger = TfBoardLogger(summary_writer)
    logger.register('reward', dtype=tf.float32)
    end_episode = lambda r, s, e: logger.plot('reward', r, s)

    def after_action(state, reward, global_step, local_step):
        if constants.LR_DECAY == 'linear':
            decay = 1.0 - (float(global_step) / constants.FINAL_STEP)
            sess.run(decay_lr_op, feed_dict={decayed_lr: constants.LR * decay})
        if global_step % 10 ** 6 == 0:
            path = os.path.join(outdir, 'model.ckpt')
            saver.save(sess, path, global_step=global_step)

    trainer = BatchTrainer(
        env=batch_env,
        agent=agent,
        render=args.render,
        state_shape=constants.IMAGE_SHAPE,
        state_window=constants.STATE_WINDOW,
        final_step=constants.FINAL_STEP,
        after_action=after_action,
        end_episode=end_episode,
        training=not args.demo
    )
    trainer.start()

if __name__ == '__main__':
    main()

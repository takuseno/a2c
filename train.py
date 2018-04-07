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
from lightsaber.rl.trainer import AsyncTrainer
from lightsaber.rl.env_wrapper import EnvWrapper
from actions import get_action_space
from network import make_network
from agent import Agent
from datetime import datetime

def make_agent(model, actions, optimizer, name):
    return Agent(
        model,
        actions,
        optimizer,
        gamma=constants.GAMMA,
        lstm_unit=constants.LSTM_UNIT,
        time_horizon=constants.TIME_HORIZON,
        policy_factor=constants.POLICY_FACTOR,
        value_factor=constants.VALUE_FACTOR,
        entropy_factor=constants.ENTROPY_FACTOR,
        grad_clip=constants.GRAD_CLIP,
        state_shape=constants.IMAGE_SHAPE + [constants.STATE_WINDOW],
        name=name
    )

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
    master = make_agent(model, actions, optimizer, 'global')

    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    saver = tf.train.Saver(global_vars)
    if args.load:
        saver.restore(sess, args.load)

    def s_preprocess(state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, tuple(constants.IMAGE_SHAPE))
        state = np.array(state, dtype=np.float32)
        return state / 255.0

    agents = []
    envs = []
    for i in range(args.threads):
        agent = make_agent(model, actions, optimizer, 'worker{}'.format(i))
        agents.append(agent)
        env = EnvWrapper(
            gym.make(args.env),
            r_preprocess=lambda r: np.clip(r, -1, 1),
            s_preprocess=s_preprocess
        )
        envs.append(env)

    initialize()

    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    logger = TfBoardLogger(summary_writer)
    logger.register('reward', dtype=tf.float32)
    end_episode = lambda r, gs, s, ge, e: logger.plot('reward', r, gs)

    def after_action(state, reward, shared_step, global_step, local_step):
        if constants.LR_DECAY == 'linear':
            decay = 1.0 - (float(shared_step) / constants.FINAL_STEP)
            sess.run(decay_lr_op, feed_dict={decayed_lr: constants.LR * decay})
        if shared_step % 10 ** 6 == 0:
            path = os.path.join(outdir, 'model.ckpt')
            saver.save(sess, path, global_step=shared_step)

    trainer = AsyncTrainer(
        envs=envs,
        agents=agents,
        render=args.render,
        state_shape=constants.IMAGE_SHAPE,
        state_window=constants.STATE_WINDOW,
        final_step=constants.FINAL_STEP,
        after_action=after_action,
        end_episode=end_episode,
        training=not args.demo,
        n_threads=args.threads
    )
    trainer.start()

if __name__ == '__main__':
    main()

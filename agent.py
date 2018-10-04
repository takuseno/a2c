import numpy as np
import tensorflow as tf

from rlsaber.util import compute_returns, compute_gae
from build_graph import build_train
from rollout import Rollout


class Agent:
    def __init__(self,
                 model,
                 actions,
                 optimizer,
                 nenvs,
                 gamma=0.99,
                 lstm_unit=256,
                 time_horizon=5,
                 value_factor=0.5,
                 entropy_factor=0.01,
                 grad_clip=40.0,
                 state_shape=[84, 84, 1],
                 phi=lambda s: s,
                 name='a2c'):
        self.actions = actions
        self.gamma = gamma
        self.name = name
        self.time_horizon = time_horizon
        self.state_shape = state_shape
        self.nenvs = nenvs
        self.phi = phi 

        self._act, self._train = build_train(
            model=model,
            num_actions=len(actions),
            optimizer=optimizer,
            nenvs=nenvs,
            step_size=time_horizon,
            lstm_unit=lstm_unit,
            state_shape=state_shape,
            grad_clip=grad_clip,
            value_factor=value_factor,
            entropy_factor=entropy_factor,
            scope=name
        )

        self.initial_state = np.zeros((nenvs, lstm_unit*2), np.float32)
        self.rnn_state = self.initial_state

        self.state_tm1 = dict(
            obs=None, action=None, value=None, done=None, rnn_state=None)
        self.rollouts = [Rollout() for _ in range(nenvs)]
        self.t = 0

    def act(self, obs_t, reward_t, done_t, training=True):
        # change state shape to WHC
        obs_t = list(map(self.phi, obs_t))

        # initialize lstm state
        for i, done in enumerate(done_t):
            if done:
                self.rnn_state[i] = self.initial_state[0]

        # take next action
        prob, value, rnn_state_t = self._act(obs_t, self.rnn_state)
        action_t = list(map(
            lambda p: np.random.choice(range(len(self.actions)), p=p), prob))
        value_t = np.reshape(value, [-1])

        if self.state_tm1['obs'] is not None:
            for i in range(self.nenvs):
                self.rollouts[i].add(
                    obs_t=self.state_tm1['obs'][i],
                    reward_tp1=reward_t[i],
                    action_t=self.state_tm1['action'][i],
                    value_t=self.state_tm1['value'][i],
                    terminal_tp1=1.0 if done_t[i] else 0.0,
                    feature_t=self.state_tm1['rnn_state'][i]
                )

        if self.t > 0 and (self.t / self.nenvs) % self.time_horizon == 0:
            # compute bootstrap value
            bootstrap_values = value_t.copy()
            for i, done in enumerate(self.state_tm1['done']):
                if done:
                    bootstrap_values[i] = 0.0
            self.train(bootstrap_values)

        self.t += self.nenvs

        self.rnn_state = rnn_state_t
        self.state_tm1['obs'] = obs_t
        self.state_tm1['action'] = action_t
        self.state_tm1['value'] = value_t
        self.state_tm1['done'] = done_t
        self.state_tm1['rnn_state'] = rnn_state_t

        return list(map(lambda a: self.actions[a], action_t))

    def train(self, bootstrap_values):
        # rollout trajectories
        obs_t,\
        actions_t,\
        rewards_tp1,\
        values_t,\
        features_t,\
        terminals_tp1,\
        masks_t = self._rollout_trajectories()

        # compute advantages
        returns_t = compute_returns(rewards_tp1, bootstrap_values,
                                    terminals_tp1, self.gamma)
        advs_t = compute_gae(rewards_tp1, values_t, bootstrap_values,
                             terminals_tp1, self.gamma, 1.0)

        # flatten inputs
        obs_t = np.reshape(obs_t, [-1] + self.state_shape)
        actions_t = np.reshape(actions_t, [-1])
        returns_t = np.reshape(returns_t, [-1])
        advs_t = np.reshape(advs_t, [-1])
        masks_t = np.reshape(masks_t, [-1])

        # train network
        loss = self._train(
            obs_t, actions_t, returns_t, advs_t, features_t, masks_t)

        # clean trajectories
        for rollout in self.rollouts:
            rollout.flush()
        return loss

    def _rollout_trajectories(self):
        obs_t = []
        actions_t = []
        rewards_tp1 = []
        values_t = []
        features_t = []
        terminals_tp1 = []
        masks_t = []
        for rollout in self.rollouts:
            obs_t.append(rollout.obs_t)
            actions_t.append(rollout.actions_t)
            rewards_tp1.append(rollout.rewards_tp1)
            values_t.append(rollout.values_t)
            terminals_tp1.append(rollout.terminals_tp1)
            features_t.append(rollout.features_t[0])
            # create mask
            mask = [0.0] + rollout.terminals_tp1[:self.time_horizon - 1]
            masks_t.append(mask)
        return obs_t, actions_t, rewards_tp1, values_t, features_t, terminals_tp1, masks_t

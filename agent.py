import numpy as np
import tensorflow as tf

from rlsaber.util import Rollout, compute_returns, compute_gae
from build_graph import build_train


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

        self.rollouts = [Rollout() for _ in range(nenvs)]
        self.t = 0

    def act(self, obs_t, reward_t, done_t, training=True):
        # change state shape to WHC
        obs_t = list(map(self.phi, obs_t))
        # take next action
        prob, value, rnn_state = self._act(obs_t, self.rnn_state)
        action_t = list(map(
            lambda p: np.random.choice(range(len(self.actions)), p=p), prob))
        value_t = np.reshape(value, [-1])

        self.t += 1
        self.rnn_state_t = self.rnn_state
        self.obs_t = obs_t
        self.action_t = action_t
        self.value_t = value_t
        self.done_t = done_t
        self.rnn_state = rnn_state
        return list(map(lambda a: self.actions[a], action_t))

    # this method is called after act
    def receive_next(self, obs_tp1, reward_tp1, done_tp1, update=False):
        obs_tp1 = list(map(self.phi, obs_tp1))

        for i in range(self.nenvs):
            self.rollouts[i].add(
                state=self.obs_t[i],
                reward=reward_tp1[i],
                action=self.action_t[i],
                value=self.value_t[i],
                terminal=1.0 if done_tp1[i] else 0.0,
                feature=self.rnn_state_t[i]
            )

        if update:
            # compute bootstrap value
            _, value, _ = self._act(obs_tp1, self.rnn_state)
            value_tp1 = np.reshape(value, [-1])
            for i, done in enumerate(done_tp1):
                if done:
                    value_tp1[i] = 0.0
            self.train(value_tp1)

        # initialize lstm state
        for i, done in enumerate(done_tp1):
            if done:
                self.rnn_state[i] = self.initial_state[0]

    def train(self, bootstrap_values):
        # rollout trajectories
        states,\
        actions,\
        rewards,\
        values,\
        features,\
        terminals,\
        masks = self._rollout_trajectories()

        # compute advantages
        returns = compute_returns(rewards, bootstrap_values, terminals, self.gamma)
        advs = compute_gae(rewards, values, bootstrap_values, terminals, self.gamma, 1.0)

        # flatten inputs
        states = np.reshape(states, [-1] + self.state_shape)
        actions = np.reshape(actions, [-1])
        returns = np.reshape(returns, [-1])
        advs = np.reshape(advs, [-1])
        masks = np.reshape(masks, [-1])

        # train network
        loss = self._train(states, actions, returns, advs, features, masks)

        # clean trajectories
        for rollout in self.rollouts:
            rollout.flush()
        return loss

    def _rollout_trajectories(self):
        states = []
        actions = []
        rewards = []
        values = []
        features = []
        terminals = []
        masks = []
        for rollout in self.rollouts:
            states.append(rollout.states)
            actions.append(rollout.actions)
            rewards.append(rollout.rewards)
            values.append(rollout.values)
            terminals.append(rollout.terminals)
            features.append(rollout.features[0])
            # create mask
            mask = [0.0] + rollout.terminals[:len(rollout.terminals) - 1]
            masks.append(mask)
        return states, actions, rewards, values, features, terminals, masks

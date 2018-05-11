from rlsaber.util import Rollout, compute_v_and_adv
from build_graph import build_train
import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self,
                 model,
                 actions,
                 optimizer,
                 nenvs,
                 gamma=0.99,
                 lstm_unit=256,
                 time_horizon=5,
                 policy_factor=1.0,
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
            lstm_unit=lstm_unit,
            state_shape=state_shape,
            grad_clip=grad_clip,
            policy_factor=policy_factor,
            value_factor=value_factor,
            entropy_factor=entropy_factor,
            scope=name
        )

        self.initial_state = np.zeros((nenvs, lstm_unit), np.float32)
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = np.zeros(([nenvs] + state_shape), dtype=np.float32)
        self.last_action = [0 for _ in range(nenvs)]
        self.last_value = [0.0 for _ in range(nenvs)]
        self.last_done = [False for _ in range(nenvs)]

        self.rollouts = [Rollout() for _ in range(nenvs)]
        self.t = 0

    def train(self):
        # get bootstrap values
        bootstrap_values = self._get_bootstrap_values()

        # rollout trajectories
        states,\
        actions,\
        rewards,\
        values,\
        features0,\
        features1,\
        masks = self._rollout_trajectories()

        # compute advantages
        targets = []
        advs = []
        for i in range(self.nenvs):
            v, adv = compute_v_and_adv(
                rewards[i], values[i], bootstrap_values[i], self.gamma)
            targets.append(v)
            advs.append(adv)

        # step size which is usually time horizon
        step_size = len(self.rollouts[0].states)

        # flatten inputs
        states = np.reshape(states, [-1] + self.state_shape)
        actions = np.reshape(actions, [-1])
        targets = np.reshape(targets, [-1])
        advs = np.reshape(advs, [-1])
        masks = np.reshape(masks, [-1])

        # train network
        loss = self._train(
            states, actions, targets, advs,
            features0, features1, masks, step_size)

        # clean trajectories
        for rollout in self.rollouts:
            rollout.flush()
        return loss

    def act(self, obs, reward, done, training=True):
        # change state shape to WHC
        obs = list(map(self.phi, obs))
        # take next action
        prob, value, rnn_state = self._act(
            obs, self.rnn_state0, self.rnn_state1)
        action = list(map(
            lambda p: np.random.choice(range(len(self.actions)), p=p), prob))
        value = np.reshape(value, [-1])

        if training:
            if self.last_obs is not None:
                for i in range(self.nenvs):
                    self.rollouts[i].add(
                        state=self.last_obs[i],
                        reward=reward[i],
                        action=self.last_action[i],
                        value=0.0 if self.last_done[i] else self.last_value[i],
                        terminal=1.0 if done[i] else 0.0,
                        feature=[self.rnn_state0[i], self.rnn_state1[i]]
                    )

        self.t += 1
        self.rnn_state0, self.rnn_state1 = rnn_state
        self.last_obs = obs
        self.last_action = action
        self.last_value = value
        self.last_done = done
        return list(map(lambda a: self.actions[a], action))

    def _rollout_trajectories(self):
        states = []
        actions = []
        rewards = []
        values = []
        features0 = []
        features1 = []
        masks = []
        for rollout in self.rollouts:
            states.append(rollout.states)
            actions.append(rollout.actions)
            rewards.append(rollout.rewards)
            values.append(rollout.values)
            features0.append(rollout.features[0][0])
            features1.append(rollout.features[0][1])
            # create mask
            terminals = rollout.terminals
            terminals = [terminals[0]] + terminals[:len(terminals) - 1]
            mask = (np.array(terminals) - 1.0) * -1.0
            masks.append(mask.tolist())
        return states, actions, rewards, values, features0, features1, masks

    def _get_bootstrap_values(self):
        values = []
        for done, value in zip(self.last_done, self.last_value):
            values.append(0.0 if done else value)
        return values

    def reset(self, index, obs):
        self.rnn_state0[index] = self.initial_state[0]
        self.rnn_state1[index] = self.initial_state[0]
        self.last_obs[index] = self.phi(obs)
        self.last_action[index] = 0
        self.last_value[index] = 0
        self.last_done[index] = False

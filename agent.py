from lightsaber.rl.util import Rollout, compute_v_and_adv
from lightsaber.rl.trainer import AgentInterface
import build_graph
import numpy as np
import tensorflow as tf


class Agent(AgentInterface):
    def __init__(self,
                 model,
                 actions,
                 optimizer,
                 gamma=0.99,
                 lstm_unit=256,
                 time_horizon=5,
                 policy_factor=1.0,
                 value_factor=0.5,
                 entropy_factor=0.01,
                 grad_clip=40.0,
                 state_shape=[84, 84, 1],
                 name='global'):
        self.actions = actions
        self.gamma = gamma
        self.name = name
        self.time_horizon = time_horizon
        self.state_shape = state_shape

        self._act,\
        self._train,\
        self._update_local,\
        self._action_dist = build_graph.build_train(
            model=model,
            num_actions=len(actions),
            optimizer=optimizer,
            lstm_unit=lstm_unit,
            state_shape=state_shape,
            grad_clip=grad_clip,
            policy_factor=policy_factor,
            value_factor=value_factor,
            entropy_factor=entropy_factor,
            scope=name
        )

        self.initial_state = np.zeros((1, lstm_unit), np.float32)
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = None
        self.last_action = None
        self.last_value = None

        self.rollout = Rollout()
        self.t = 0

    def train(self, bootstrap_value):
        states = np.array(self.rollout.states, dtype=np.float32)
        actions = np.array(self.rollout.actions, dtype=np.uint8)
        rewards = self.rollout.rewards
        values = self.rollout.values
        v, adv = compute_v_and_adv(rewards, values, bootstrap_value, self.gamma)
        loss = self._train(
            states, self.rollout.features[0][0], self.rollout.features[0][1],
            actions, v, adv)
        self._update_local()
        return loss

    def act(self, obs, reward, training=True):
        # change state shape to WHC
        obs = np.transpose(obs, [1, 2, 0])
        # take next action
        prob, value, rnn_state = self._act(
            [obs], self.rnn_state0, self.rnn_state1)
        action = np.random.choice(range(len(self.actions)), p=prob[0])

        if training:
            if len(self.rollout.states) == self.time_horizon:
                self.train(self.last_value)
                self.rollout.flush()

            if self.last_obs is not None:
                self.rollout.add(
                    state=self.last_obs,
                    reward=reward,
                    action=self.last_action,
                    value=self.last_value,
                    terminal=False,
                    feature=[self.rnn_state0, self.rnn_state1]
                )

        self.t += 1
        self.rnn_state0, self.rnn_state1 = rnn_state
        self.last_obs = obs
        self.last_action = action
        self.last_value = value[0][0]
        return self.actions[action]

    def stop_episode(self, obs, reward, training=True):
        if training:
            self.rollout.add(
                state=self.last_obs,
                action=self.last_action,
                reward=reward,
                value=self.last_value,
                terminal=True
            )
            self.train(0)
            self.rollout.flush()
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = None
        self.last_action = None
        self.last_value = None

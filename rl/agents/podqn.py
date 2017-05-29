from __future__ import division
import warnings

import keras.backend as K
from keras.layers import Lambda, Input, Layer, Dense

from rl.core import Agent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import *
from rl.keras_future import Model


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class AbstractDQNAgent(Agent):
    """Write me
    """
    def __init__(self, nb_actions, memory, gamma=.99, batch_size=32, nb_steps_warmup=1000,
                 train_interval=1, memory_interval=1, target_model_update=10000,
                 delta_range=None, delta_clip=np.inf, custom_model_objects={}, **kwargs):
        super(AbstractDQNAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.memory = memory

        # State.
        self.compiled = False

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def compute_batch_q_values(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(batch)
        assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state]).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def get_config(self):
        return {
            'nb_actions': self.nb_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_clip': self.delta_clip,
            'memory': get_object_config(self.memory),
        }


class PODQNAgent(AbstractDQNAgent):
    """Write me
    """
    def __init__(self, V_model, L_model, mu_model, random_process=None,
                 covariance_mode='full', *args, **kwargs):
        super(NAFAgent, self).__init__(*args, **kwargs)

        # TODO: Validate (important) input.

        # Parameters.
        self.random_process = random_process
        self.covariance_mode = covariance_mode

        # Related objects.
        self.V_model = V_model
        self.L_model = L_model
        self.mu_model = mu_model

        # State.
        self.reset_states()

    def update_target_model_hard(self):
        self.target_V_model.set_weights(self.V_model.get_weights())

    def load_weights(self, filepath):
        self.combined_model.load_weights(filepath)  # updates V, L and mu model since the weights are shared
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.combined_model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.combined_model.reset_states()
            self.target_V_model.reset_states()

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # Create target V model. We don't need targets for mu or L.
        self.target_V_model = clone_model(self.V_model, self.custom_model_objects)
        self.target_V_model.compile(optimizer='sgd', loss='mse')

        # Build combined model.
        a_in = Input(shape=(self.nb_actions,), name='action_input')
        if type(self.V_model.input) is list:
            observation_shapes = [i._keras_shape[1:] for i in self.V_model.input]
        else:
            observation_shapes = [self.V_model.input._keras_shape[1:]]
        os_in = [Input(shape=shape, name='observation_input_{}'.format(idx)) for idx, shape in enumerate(observation_shapes)]
        L_out = self.L_model([a_in] + os_in)
        V_out = self.V_model(os_in)

        mu_out = self.mu_model(os_in)
        A_out = NAFLayer(self.nb_actions, mode=self.covariance_mode)([L_out, mu_out, a_in])
        combined_out = Lambda(lambda x: x[0]+x[1], output_shape=lambda x: x[0])([A_out, V_out])
        combined = Model(input=[a_in] + os_in, output=[combined_out])
        # Compile combined model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_V_model, self.V_model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        combined.compile(loss=clipped_error, optimizer=optimizer, metrics=metrics)
        self.combined_model = combined

        self.compiled = True

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.mu_model.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            # Compute Q values for mini-batch update.
            q_batch = self.target_V_model.predict_on_batch(state1_batch).flatten()
            assert q_batch.shape == (self.batch_size,)

            # Compute discounted reward.
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            assert Rs.shape == (self.batch_size,)

            # Finally, perform a single update on the entire batch.
            if len(self.combined_model.input) == 2:
                metrics = self.combined_model.train_on_batch([action_batch, state0_batch], Rs)
            else:
                metrics = self.combined_model.train_on_batch([action_batch] + state0_batch, Rs)
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def layers(self):
        return self.combined_model.layers[:]

    def get_config(self):
        config = super(NAFAgent, self).get_config()
        config['V_model'] = get_object_config(self.V_model)
        config['mu_model'] = get_object_config(self.mu_model)
        config['L_model'] = get_object_config(self.L_model)
        if self.compiled:
            config['target_V_model'] = get_object_config(self.target_V_model)
        return config

    @property
    def metrics_names(self):
        names = self.combined_model.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names


# Aliases
ContinuousDQNAgent = NAFAgent

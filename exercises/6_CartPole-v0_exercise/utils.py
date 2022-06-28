import gym
import numpy as np
import random
from sklearn.exceptions import NotFittedError
from tensorflow.keras import models as M
from tensorflow.keras import layers as L
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks as C

ENV = gym.make('CartPole-v0')


class QDNN:
    """The state action value function."""
    def __init__(self, model=None):

        # The action space has dimension 1.
        self.input_dim = ENV.observation_space.shape[0] + 1

        if model is None:
            model = M.Sequential()
            model.add(L.Input(shape=(self.input_dim,)))
            model.add(L.Normalization(name='normalization'))
            model.add(L.Dense(24, activation='relu'))
            model.add(L.Dense(24, activation='relu'))
            model.add(L.Dense(24, activation='relu'))
            model.add(L.Dense(1, activation='relu'))
            model.add(L.Activation('linear'))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mse']
            )
        self.normalization = model.get_layer('normalization')
        self.model = model

    def __call__(self, state, action):
        try:
            return self.model(np.r_[state, action].reshape(1, -1), training=False).numpy().flatten()
        except NotFittedError:
            return np.random.randn(1, 1)

    def policy(self, state):
        """Epsilon greedy policy using the value approximation."""
        return np.argmax([self(state, action)
            for action in range(ENV.action_space.n)
        ])

    @classmethod
    def from_file(cls, path):
        return cls(model=M.load_model(path))


def normalization_adapt(Q, experience):
    Xs = []
    for state, action, _, _ in experience:

        # The inputs to the state action value function.
        X = np.r_[state, action]

        # Save.
        Xs.append(X)

    X = np.array(Xs)
    Q.normalization.adapt(X)


def replay(Q, experience, gamma, batch_size, epochs, training_size):
    """Experience replay. Fit based on cached experience."""

    # The time series has a high degree of autocorrelation across each episode.
    # Taking a random sample decreases this correlation/dependence.
    # This is the main idea of experience replay.
    minibatch = random.choices(experience, k=training_size)

    Xs = []
    ys = []
    for state, action, reward, new_state in minibatch:

        # The inputs to the state action value function.
        X = np.r_[state, action]

        # The expected return (expected discounted return/reward) starting from s, taking the action a, and thereafter following policy π.
        if reward == -1:
            y = np.array([reward])
        else:
            y = reward + gamma * max(Q(new_state, action) for action in range(ENV.action_space.n))

        # Save.
        Xs.append(X)
        ys.append(y)

    X = np.array(Xs)
    y = np.array(ys)
    Q.model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[
            C.TensorBoard('./logs', update_freq=1),
            C.EarlyStopping(patience=2),
            C.ModelCheckpoint(
                filepath='best_model.h5',
                save_weights_only=False,
                monitor='val_mse',
                mode='max',
                save_best_only=True
            )
        ]
    )


def run_episode(Q, epsilon=0, render=False):
    """Generate an episode."""
    experience = []
    try:
        state = ENV.reset()
        while True:

            if render:
                ENV.render()

            # With probability epsilon take a random action.
            if epsilon < 1:
                action = Q.policy(state)
            if random.random() <= epsilon:
                action = ENV.action_space.sample()

            new_state, reward, done, _ = ENV.step(action)
            if done:
                reward = -1

            # Save SARS.
            experience.append((state, action, reward, new_state))

            if done:
                break

            # Current state/reward.
            state = new_state
    finally:
        ENV.close()
    return experience

import random
import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import utils


class ReservoirList:
    """A list keeping a resevoir sample using the resevoir sampling algorithm."""

    def __init__(self, resevoir_size):
        self.resevoir_size = resevoir_size
        self._list = []
        self._i = 0

    def sample(self, k):
        return self._list[0:k]

    def append(self, item):
        j = random.randint(0, self._i)
        if j <= self.resevoir_size - 1:
            self._list[j] = item
        self._i += 1

    def extend(self, items):
        items_size = len(items)
        if len(self._list) < self.resevoir_size:
            if len(self._list) + len(items) < self.resevoir_size:
                self._list.extend(items)
            else:
                while len(self._list) < self.resevoir_size:
                    item = items.pop()
                    self._list.append(item)
                for item in items:
                    self.append(item)
        else:
            for item in items:
                self.append(item)
        self._i += items_size

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._list)

    def __repr__(self):
        return f'ReservoirSampleList(resevoir_size={self.resevoir_size}, _i={self.i}, _list={self._list})'


def main(
    render=False,
    gamma=0.95,
    epsilon=0.1,
    n_episodes=500,
    training_size=10000,
    experience_size=1000000,
    batch_size=64,
    epochs=50,
):
    Q = utils.QDNN()
    Q.model.summary()
    experience = ReservoirList(resevoir_size=experience_size)
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(log_dir + '/reward')
    while len(experience) < experience_size:
        print(f"\n--- Filling up experience ---")
        episode = utils.run_episode(Q, epsilon=1, render=False)
        print('Episode length', len(episode))
        print('Episode reward', sum(reward for _, _, reward, _ in episode))
        print('Experience length', len(experience))
        experience.extend(episode)
    with writer.as_default():
        tf.summary.scalar("episode_reward", sum(reward for _, _, reward, _ in episode), step=0)
    for episode_idx in range(n_episodes):
        print(f"\n--- Replay {episode_idx} ---")
        utils.normalization_adapt(Q, experience)
        utils.replay(
            Q, experience, gamma, batch_size, training_size,
            initial_epoch=episode_idx * epochs,
            epochs=(episode_idx + 1) * epochs,
            log_dir=log_dir,
        )
        episode = utils.run_episode(Q, epsilon=epsilon, render=render)
        print('Episode length', len(episode))
        print('Episode reward', sum(reward for _, _, reward, _ in episode))
        print('Experience length', len(experience))
        experience.extend(episode)
        with writer.as_default():
            tf.summary.scalar("episode_reward", sum(reward for _, _, reward, _ in episode), step=(episode_idx + 1) * epochs)
        episode_idx += 1


if __name__ == '__main__':
    main()

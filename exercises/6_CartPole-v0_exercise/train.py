import collections

import utils


def main(
    render=False,
    gamma=0.95,
    epsilon=0.1,
    n_episodes=500,
    training_size=1000,
    experience_size=100000,
    batch_size=64,
    epochs=20,
):
    Q = utils.QDNN()
    Q.model.summary()
    experience = collections.deque(maxlen=experience_size)
    while len(experience) < experience_size:
        print(f"\n--- Filling up experience ---")
        episode = utils.run_episode(Q, epsilon=1, render=False)
        print('Episode length', len(episode))
        print('Episode reward', sum(reward for _, _, reward, _ in episode))
        print('Experience length', len(experience))
        experience.extend(episode)
    for episode_idx in range(n_episodes):
        print(f"\n--- Replay {episode_idx} ---")
        utils.normalization_adapt(Q, experience)
        utils.replay(Q, experience, gamma, batch_size, epochs, training_size)
        episode = utils.run_episode(Q, epsilon=epsilon, render=render)
        print('Episode length', len(episode))
        print('Episode reward', sum(reward for _, _, reward, _ in episode))
        print('Experience length', len(experience))
        experience.extend(episode)


if __name__ == '__main__':
    main()

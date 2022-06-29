import utils
from tensorflow import keras


def main():
    Q = utils.QDNN.from_file('best_model.h5')
    while True:
        utils.run_episode(Q, epsilon=0, render=True)


if __name__ == '__main__':
    main()

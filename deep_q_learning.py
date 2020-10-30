import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses


class Model(models.Sequential):
    def __init__(self):
        # TODO try 2d convolution
        super().__init__()
        self.add(keras.Input(shape=42))
        self.add(layers.Dense(20, activation='relu'))
        self.add(layers.Dense(40, activation='relu'))
        self.add(layers.Dense(7, activation='sigmoid'))  # map to (0,1)  # TODO try other activations?
        self.compile(
            loss=losses.MeanSquaredError(),  # TODO play around with loss functions; https://stats.stackexchange.com/a/234578
            optimizer=optimizers.Adam(lr=0.0001)  # TODO play around with optimizers
        )


class Agent:
    def __init__(self, model, exploration_rate=0.75):
        self.model = model  # the brain itself
        self.exploration_rate = exploration_rate

    def make_move(self, board_state):
        """
        output is 0-6, for which column to use for this action
        """
        if random.random() < exploration_rate:
            return np.random.choice(7)  # explore! choose randomly 0-6 inclusive
        # TODO use brainz
        raise Exception("not implemented")

    def process_feedback(self, from_state, chosen_move, reward, game_over):
        """
        game_end: aka end of the _episode_; i.e. a win, loss, or draw
        """
        # TODO until the game is over, store the feedback as a tuple
        #  once the game is over, process the feedback & train the model (in batches?)
        raise Exception("not implemented")


# THE ENVIRONMENT FOR THE AGENT
if __name__ == "__main__":
    storage_path = "./deep_q_learning_weights.h5"
    episodes = 1000

    exploration_rate = 0.75  # aka epsilon
    discount_factor = 0.9  # aka gamma

    # TODO


import random
from collections import deque

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
        #  or otherwise try regularization
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
    def __init__(self, model, exploration_rate, discount_factor):
        self.model = model  # the brain itself
        self.exploration_rate = exploration_rate
        self.discount_factor = discount_factor
        self.training_data = deque(maxlen=500)  # store last 500 moves

    def make_move(self, state_array):
        """
        output is 0-6, for which column to use for this action
        """
        if random.random() < exploration_rate:
            return np.random.choice(7)  # explore! choose randomly 0-6 inclusive
        predictions = self.model.predict(state_array)
        return np.argmax(predictions[0])  # pick the column with the highest value

    def update_model(self):
        batch_size = 100
        if len(self.training_data) < batch_size:
            return
        batch = random.sample(self.training_data, batch_size)
        for from_state_array, chosen_move, to_state_array, reward, game_over in batch:
            # if the game is over, there are no valid predictions to be made from the game-ending state
            # but otherwise, we consider the prediction from the following state
            target = reward if game_over else \
                reward + self.discount_factor * np.amax(self.model.predict(to_state_array)[0])
            # for the training labels, use model predictions with changed "chosen_move" value
            labels = self.model.predict(from_state_array)
            labels[0][chosen_move] = target
            self.model.fit(from_state_array, labels, epochs=1, verbose=0)

    def process_feedback(self, from_state_array, chosen_move, to_state_array, reward, game_over):
        """
        game_end: aka end of the _episode_; i.e. a win, loss, or draw
        """
        self.training_data.append((from_state_array, chosen_move, to_state_array, reward, game_over))


# THE ENVIRONMENT FOR THE AGENT
if __name__ == "__main__":
    weights_storage_path = "./deep_q_learning_weights.h5"
    episodes = 1000

    exploration_rate = 0.75  # aka epsilon
    discount_factor = 0.9  # aka gamma

    model = Model()
    model.load_weights(weights_storage_path)

    agent = Agent(model, exploration_rate, discount_factor)

    # TODO run episodes


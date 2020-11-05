import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses

from connect4_env import GameBoard, PLAYER1, PLAYER2


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
        self.exploration_rate = exploration_rate  # TODO epsilon decay
        self.discount_factor = discount_factor
        self.training_data = deque(maxlen=500)  # store last 500 moves

    def make_move(self, state_array):
        """
        output is 0-6, for which column to use for this action
        """
        if random.random() < self.exploration_rate:
            return np.random.choice(7)  # explore! choose randomly 0-6 inclusive
        predictions = self.model.predict(state_array)
        return np.argmax(predictions[0])  # pick the column with the highest value

    def update_model(self):
        batch_size = 100
        if len(self.training_data) < batch_size:
            return
        batch = random.sample(self.training_data, batch_size)
        # TODO try iterating backwards!
        #  doing that once per episode might update more effectively; would this be a monte carlo approach vs TD?
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
def run():
    weights_storage_path = "./deep_q_learning_weights.h5"
    episodes = 1000

    exploration_rate = 0.75  # aka epsilon
    discount_factor = 0.9  # aka gamma

    model = Model()
    model.load_weights(weights_storage_path)

    agent = Agent(model, exploration_rate, discount_factor)

    for episode in range(episodes):
        board = GameBoard(next_player=random.choice([PLAYER1, PLAYER2]))  # new game!

        if board.get_next_player() == PLAYER2:
            board.make_move(random.choice(board.get_available_columns()))  # start with a random opponent move

        while True:
            from_state = board.to_array()

            move = agent.make_move(from_state)
            if not board.is_column_available(move):  # invalid move! punish agent & end episode
                agent.process_feedback(from_state, move, None, -100, True)  # very strong penalty
                break
            board.make_move(move)  # agent move

            to_state = board.to_array()

            if board.has_won(PLAYER1):
                agent.process_feedback(from_state, move, None, 100, True)
                break  # win - end episode

            available_columns = board.get_available_columns()
            if len(available_columns) > 0:
                board.make_move(random.choice(board.get_available_columns()))  # random opponent move
                available_columns = board.get_available_columns()

            if board.has_won(PLAYER2):
                agent.process_feedback(from_state, move, None, 100, True)
                break  # loss - end episode

            agent.process_feedback(from_state, move, to_state, 1, True)

            if len(available_columns) == 0:
                break  # draw - end episode


if __name__ == "__main__":
    run()

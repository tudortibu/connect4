import os
import random
from collections import deque
from time import time

import numpy as np
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses

from connect4_env import GameBoard, PLAYER1, PLAYER2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # uncomment this to disable gpu


class Model(models.Sequential):
    def __init__(self):
        # TODO try 2d convolution
        #  or otherwise try regularization
        super().__init__()
        self.add(keras.Input(shape=42))
        self.add(layers.Dense(50, activation='relu'))
        self.add(layers.Dense(50, activation='relu'))
        self.add(layers.Dense(50, activation='relu'))
        self.add(layers.Dense(50, activation='relu'))
        self.add(layers.Dense(21, activation='relu'))
        self.add(layers.Dense(7))
        self.compile(
            loss=losses.MeanSquaredError(),  # TODO play around with loss functions; https://stats.stackexchange.com/a/234578
            optimizer=optimizers.Adam(lr=0.0001)  # TODO play around with optimizers % learning rate
        )


class Agent:
    def __init__(self, model, discount_factor):
        self.model = model  # the brain itself
        self.discount_factor = discount_factor

        self.memory_NT = deque(maxlen=2000)  # last 2000 non-terminal transitions
        self.memory_T = deque(maxlen=1000)  # last 1000 terminal transitions

    def choose_move(self, state_array, exploration_rate):
        """
        output is 0-6, for which column to use for this action
        """
        if random.random() < exploration_rate:
            return np.random.choice(7)  # explore! choose randomly 0-6 inclusive
        predictions = self.model.predict(np.array([state_array]))
        return np.argmax(predictions[0])  # pick the column with the highest value

    def learn(self):
        self.learn_on(True, 100)  # less (but episode-proportionally more) terminal data
        self.learn_on(False, 400)  # more non-terminal data

    def learn_on(self, terminal_data, batch_size):
        data = self.memory_T if terminal_data else self.memory_NT
        if len(data) < batch_size:
            return
        batch = random.sample(data, batch_size)

        from_states = np.zeros((batch_size, 42))
        moves = np.zeros(batch_size, dtype=np.int8)  # integers, [0-7)
        to_states = np.zeros((batch_size, 42))  # unused if terminal_data == True
        rewards = np.zeros(batch_size)

        for i in range(batch_size):
            from_state, move, to_state, reward = batch[i]
            from_states[i] = from_state
            moves[i] = move
            to_states[i] = to_state
            rewards[i] = reward

        labels = self.model.predict(from_states)  # (batch_size, 7)
        # skipping the following calculation for terminal data, as it's not needed then
        to_state_predictions = None if terminal_data else self.model.predict(to_states)

        for i in range(batch_size):
            labels[i, moves[i]] = rewards[i] +\
                               (0 if terminal_data else self.discount_factor * np.amax(to_state_predictions[i]))

        self.model.fit(from_states, labels, epochs=1, verbose=0, batch_size=batch_size, shuffle=False)

        # OLD LOGIC
        # for from_state_array, chosen_move, to_state_array, reward, game_over in batch:
        #     # if the game is over, there are no valid predictions to be made from the game-ending state
        #     # but otherwise, we consider the prediction from the following state
        #     target = reward if game_over else \
        #         reward + self.discount_factor * np.amax(self.model.predict(to_state_array)[0])
        #     # for the training labels, use model predictions with changed "chosen_move" value
        #     labels = self.model.predict(from_state_array)
        #     labels[0][chosen_move] = target
        #     self.model.fit(from_state_array, labels, epochs=1, verbose=0)

    def process_feedback(self, from_state_array, chosen_move, to_state_array, reward, game_over):
        """
        game_end: aka end of the _episode_; i.e. a win, loss, or draw
        """
        from_state = np.array([from_state_array])
        to_state = np.array([to_state_array]) if to_state_array is not None else None
        data = (from_state, chosen_move, to_state, reward)
        self.memory_T.append(data) if game_over else self.memory_NT.append(data)


def get_exploration_rate(episode):
    return 0.1
    return max(0.05, 0.9995 ** episode)  # gives flat .05 past 6,000 episodes


# THE ENVIRONMENT FOR THE AGENT
def run():
    weights_storage_path = "./deep_q_learning_weights.h5"
    episodes = 20000

    discount_factor = 0.9  # aka gamma

    model = Model()
    if os.path.isfile(weights_storage_path):
        model.load_weights(weights_storage_path)

    agent = Agent(model, discount_factor)

    average_win_rate = 0
    average_reward = 0
    average_moves = 0
    average_invalid_move_rate = 0
    last_verbose_epoch = int(time())

    start_episode = 0
    for episode in range(start_episode, episodes+1, 1):
        board = GameBoard(next_player=random.choice([PLAYER1, PLAYER2]))  # new game!

        exploration_rate = get_exploration_rate(episode)

        # play an episode
        total_reward, invalid_move = play_against_self(agent, board, exploration_rate)
        # have the agent learn a little
        agent.learn()

        average_win_rate *= 0.99
        average_win_rate += 0.01 if board.has_won(PLAYER1) else 0

        average_reward *= 0.99
        average_reward += 0.01*total_reward

        average_moves *= 0.99
        average_moves += 0.01 * board.total_moves/2

        average_invalid_move_rate *= 0.99
        average_invalid_move_rate += 0.01 if invalid_move else 0

        if episode % 10 == 0:
            model.save_weights(weights_storage_path)
            board.print()
            now = int(time())
            time_delta = now-last_verbose_epoch
            last_verbose_epoch = now
            print("episode #%d - win rate: %d%% - epsilon: %.2f - avg reward: %d - avg moves: %d - invalid rate: %d%% - delta time: %dsec" %
                  (episode,
                   average_win_rate*100,
                   exploration_rate,
                   average_reward,
                   average_moves,
                   average_invalid_move_rate*100,
                   time_delta
                   )
                  )


def play_against_random(agent, board, exploration_rate):

    total_reward = 0
    invalid_move = False

    last_agent_from_state = None
    last_agent_move = None

    while len(board.get_available_columns()) > 0:
        # OPPONENT'S TURN
        if board.get_next_player() == PLAYER2:
            move = random.choice(board.get_available_columns())
            board.make_move(move)  # opponent makes random move
            if board.has_won(PLAYER2):
                agent.process_feedback(last_agent_from_state, last_agent_move, None, -100, True)
                total_reward -= 100
                # also, tell the agent that blocking the win would've been a good move!
                if move != last_agent_move:
                    board = board.copy()
                    board.undo_move()
                    board.undo_move()
                    from_state = board.to_array()
                    board.make_move(move)
                    if board.has_won(PLAYER1):  # even better, this move would've given a win!
                        agent.process_feedback(from_state, move, board.to_array(), 100, True)
                    else:  # otherwise, give a good reward for the theoretical blocking of the opponent's win
                        agent.process_feedback(from_state, move, board.to_array(), 20, False)
                break  # loss - end episode
        # AGENT'S TURN
        else:
            from_state = board.to_array()
            move = agent.choose_move(from_state, exploration_rate)
            reward = 1  # default reward for generic move
            game_over = False

            if not board.is_column_available(move):
                reward = -300  # invalid move! hard penalty
                game_over = True
                invalid_move = True
            else:
                board.make_move(move)  # move is valid, do it
                if board.has_won(PLAYER1):
                    reward = 100  # win! significant reward
                    game_over = True

            agent.process_feedback(from_state, move, board.to_array(), reward, game_over)

            last_agent_from_state = from_state
            last_agent_move = move
            total_reward += reward

            if game_over:
                break

    return total_reward, invalid_move


def play_against_self(agent, board, exploration_rate):

    total_reward = {PLAYER1: 0, PLAYER2: 0}
    invalid_move = False

    last_move = None

    while len(board.get_available_columns()) > 0:
        currently_playing_as = board.get_next_player()

        from_state = board.to_array(perspective=currently_playing_as)
        move = agent.choose_move(from_state, exploration_rate)
        reward = 1  # default reward for generic move
        game_over = False
        to_state = None

        if not board.is_column_available(move):
            reward = -300  # invalid move! hard penalty
            game_over = True
            invalid_move = True
        else:
            board.make_move(move)  # move is valid, do it
            to_state = board.to_array(perspective=currently_playing_as)
            if board.has_won(currently_playing_as):
                reward = 100  # win! significant reward
                game_over = True
                # also, tell the agent, from opponent perspective, that blocking this win would've been a good move
                if move != last_move:
                    other_player = PLAYER2 if currently_playing_as == PLAYER1 else PLAYER1
                    board_copy = board.copy()
                    board_copy.undo_move()  # undo this win
                    board_copy.undo_move()  # undo whatever the opponent did (last_move)
                    from_state2 = board_copy.to_array(perspective=other_player)
                    board_copy.make_move(move)  # assume opponent chose this winning column instead
                    to_state2 = board_copy.to_array(perspective=other_player)
                    if board_copy.has_won(other_player):  # opponent would have won!
                        agent.process_feedback(from_state2, move, None, 100, True)
                    else:  # opponent would have blocked the win, which is way better than making a non-winning move
                        agent.process_feedback(from_state2, move, to_state2, 20, False)

        agent.process_feedback(from_state, move, None if game_over else to_state, reward, game_over)

        last_move = move
        total_reward[currently_playing_as] += reward

        if game_over:
            break

    avg_reward = (total_reward[PLAYER1]+total_reward[PLAYER2])/2
    return avg_reward, invalid_move


if __name__ == "__main__":
    run()

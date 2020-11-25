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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # comment this to enable gpu


class Model(models.Sequential):
    def __init__(self):
        # TODO try 2d convolution
        #  or otherwise try regularization
        super().__init__()
        self.add(layers.Dense(42, input_dim=42, activation='relu'))
        self.add(layers.Dense(42, activation='relu'))
        self.add(layers.Dense(42, activation='relu'))
        self.add(layers.Dense(42, activation='relu'))
        self.add(layers.Dense(21, activation='relu'))
        self.add(layers.Dense(7))
        self.compile(
            loss=losses.MeanSquaredError(),  # TODO play around with loss functions; https://stats.stackexchange.com/a/234578
            optimizer=optimizers.Adam(lr=0.00001)  # TODO play around with optimizers % learning rate
        )
        self.summary()


class ConvolutionalModel(models.Sequential):
    def __init__(self):
        super().__init__()
        self.add(layers.Reshape((7, 6, 1), input_dim=42))  # channels_last

        self.add(layers.Conv2D(64, kernel_size=(5, 5), input_shape=(7, 6, 1), activation='relu'))
        # self.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(layers.Dropout(rate=0.2))

        # TODO diverge & combine multiple parallel convolution layers using functional api?

        # self.add(layers.Conv2D(21, kernel_size=(4, 4), input_shape=(7, 6, 1), activation='relu'))
        # self.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # self.add(layers.Dropout(rate=0.15))

        self.add(layers.Flatten())
        self.add(layers.Dense(42, activation='relu'))
        self.add(layers.Dense(21, activation='relu'))
        self.add(layers.Dense(7))

        self.compile(
            loss=losses.MeanSquaredError(),
            optimizer=optimizers.Adam(lr=0.00001)
        )
        self.summary()


class Agent:
    def __init__(self, model, discount_factor):
        self.model = model  # the brain itself
        self.discount_factor = discount_factor
        self.feedback = deque(maxlen=50)  # TODO try contracting/expanding this feedback buffer & evaluate affects?

    def choose_move(self, state_array, exploration_rate):
        """
        output is 0-6, for which column to use for this action
        """
        if random.random() < exploration_rate:
            return np.random.choice(7)  # explore! choose randomly 0-6 inclusive
        predictions = self.model.predict(np.array([state_array]))
        return np.argmax(predictions[0])  # pick the column with the highest value

    def learn(self):
        batch_size = 30
        if len(self.feedback) < batch_size:
            return 0

        batch = random.sample(self.feedback, batch_size)

        total_loss = 0

        for from_state, move, to_state, reward in batch:
            fit_input = np.array([from_state])
            predictions = self.model.predict(fit_input if to_state is None else np.array([from_state, to_state]))
            labels = predictions[0]
            labels[move] = reward
            if to_state is not None:  # not terminal move; add discounted reward from subsequent state
                labels[move] += self.discount_factor * np.amax(predictions[1])
            total_loss += self.model.fit(fit_input, np.array([labels]), epochs=1, verbose=0, shuffle=False).history["loss"][0]

        return total_loss/batch_size

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

    def process_feedback(self, from_state_array, chosen_move, to_state_array, reward):
        """
        game_end: aka end of the _episode_; i.e. a win, loss, or draw
        """
        self.feedback.append((from_state_array, chosen_move, to_state_array, reward))


def get_exploration_rate(episode):
    return max(0.05, 0.9995 ** episode)  # gives flat .05 past 6,000 episodes


# THE ENVIRONMENT FOR THE AGENT
def run():
    weights_storage_path = "weights.h5"
    episodes = 50000

    discount_factor = 0.9  # gamma

    model = ConvolutionalModel()  # TODO take note of which model is being used
    if os.path.isfile(weights_storage_path):
        model.load_weights(weights_storage_path)

    agent = Agent(model, discount_factor)

    stat_batch_size = 10

    stat_file = "stats.npy"
    stat_data = np.zeros((int(episodes/stat_batch_size), 6))
    if os.path.isfile(stat_file):
        stat_data = np.load(stat_file)
    stat_temp = np.zeros(6)  # loss, win_rate, reward, moves, invalid_rate, draw_rate

    last_verbose_epoch = int(time())

    start_episode = 1
    for episode in range(start_episode, episodes+1, 1):
        board = GameBoard(next_player=random.choice([PLAYER1, PLAYER2]))  # new game!

        exploration_rate = get_exploration_rate(episode)

        # play an episode
        total_reward, invalid_move = play_against_random(agent, board, exploration_rate)

        stat_temp[0] += agent.learn()
        stat_temp[1] += 1 if board.has_won(PLAYER1) else 0
        stat_temp[2] += total_reward
        stat_temp[3] += board.total_moves/2
        stat_temp[4] += 1 if invalid_move else 0
        stat_temp[5] += 1 if not invalid_move and board.total_moves == 42 else 0

        if episode % stat_batch_size == 0:
            model.save_weights(weights_storage_path)

            stat_temp /= stat_batch_size
            stat_index = int(episode / stat_batch_size)-1  # batch indexing starts at 0
            stat_data[stat_index] = stat_temp
            stat_temp = np.zeros(6)
            np.save(stat_file, stat_data)

            now = int(time())
            time_delta = now - last_verbose_epoch
            last_verbose_epoch = now

            board.print()  # show the current EP board state

            print("EP %d --- loss: %d, win: %d%%, reward: %d, moves: %d, invalid: %d%% --- time: %dsec" % (
                episode,
                stat_data[stat_index][0],
                stat_data[stat_index][1]*100,
                stat_data[stat_index][2],
                stat_data[stat_index][3],
                stat_data[stat_index][4]*100,
                time_delta
            ))


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
                agent.process_feedback(last_agent_from_state, last_agent_move, None, -100)
                total_reward -= 100

                # also, tell the agent that blocking the win would've been a good move!
                if move != last_agent_move:
                    board = board.copy()
                    board.undo_move()  # undo PLAYER2 win
                    board.undo_move()  # undo whatever the agent did (last_agent_move)
                    from_state = board.to_array()
                    board.make_move(move)
                    to_state = board.to_array()
                    if board.has_won(PLAYER1):  # even better, this move would've given a win!
                        agent.process_feedback(from_state, move, None, 100)
                    else:  # otherwise, give a good reward for the theoretical blocking of the opponent's win
                        agent.process_feedback(from_state, move, to_state, 95)

                break  # loss - end episode
        # AGENT'S TURN
        else:
            from_state = board.to_array()
            move = agent.choose_move(from_state, exploration_rate)
            reward = -5  # default reward for generic move; discourages filling up the board to get small rewards  # TODO make a training instance on this = -1
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

            agent.process_feedback(from_state, move, None if game_over else board.to_array(), reward)

            last_agent_from_state = from_state
            last_agent_move = move
            total_reward += reward

            if game_over:
                break

    return total_reward, invalid_move


def play_against_smart_adversary(agent, board, exploration_rate):

    total_reward = 0
    invalid_move = False

    while len(board.get_available_columns()) > 0:
        # ADVERSARY'S TURN
        if board.get_next_player() == PLAYER2:
            move = random.choice(board.get_available_columns())

            # smart logic
            if board.total_moves >= 6:
                blocking_move_check_count = 0

                for test_move in board.get_available_columns():
                    board.make_move(test_move)  # test this move

                    if board.has_won(PLAYER2):
                        move = test_move  # make this winning move (1st priority)
                        board.undo_move()
                        break

                    # there is no point to run the following nested loop more than twice
                    if blocking_move_check_count < 2:
                        blocking_move_check_count += 1
                        for test_agent_move in board.get_available_columns():
                            if test_move != test_agent_move:
                                board.make_move(test_agent_move)  # assume agent makes this move

                                if board.has_won(PLAYER1):  # the agent would win!
                                    move = test_agent_move  # block the agent from winning (2nd priority)
                                    board.undo_move()
                                    break

                                board.undo_move()

                    board.undo_move()

            board.make_move(move)
            if board.has_won(PLAYER2):
                break  # loss - end episode
        # AGENT'S TURN
        else:
            from_state = board.to_array()
            move = agent.choose_move(from_state, exploration_rate)
            reward = -1  # default reward for generic move
            game_over = False

            # help the agent learn to win/block in the future; give feedback on missed opportunities
            for test_move in board.get_available_columns():
                if test_move != move:
                    # test for blocking move
                    if board.is_blocking_move(test_move):
                        board.make_move(test_move)
                        agent.process_feedback(from_state, test_move, board.to_array(), 95)
                        board.undo_move()
                    # test for winning move
                    else:
                        board.make_move(test_move)
                        if board.has_won(PLAYER1):
                            agent.process_feedback(from_state, test_move, None, 100)
                        board.undo_move()

            if not board.is_column_available(move):
                reward = -200  # invalid move! hard penalty
                game_over = True
                invalid_move = True
            else:
                board.make_move(move)  # move is valid, do it
                if board.has_won(PLAYER1):
                    reward = 100  # win! significant reward
                    game_over = True
                else:
                    # check for losing/bad move
                    for test_move in board.get_available_columns():
                        board.make_move(test_move)
                        if board.has_won(PLAYER2):
                            reward = -100  # the agent's move allows the adversary to win; give high penalty
                            board.undo_move()
                            break
                        board.undo_move()
                    # check for blocking/good move (only if not losing move)
                    if reward != -100 and len(board.get_available_columns()) > 0:
                        board.undo_move()  # temporarily undo the move
                        if board.is_blocking_move(move):
                            reward = 95  # the agent blocked the adversary from winning; give high reward
                        board.make_move(move)  # restore the move

            agent.process_feedback(from_state, move, None if game_over else board.to_array(), reward)

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

        # to simplify initial learning, pardon a good % of invalid moves (give agent more chance to explore)
        invalid_attempt_count = 0
        while True:
            move = agent.choose_move(from_state, exploration_rate)
            if not board.is_column_available(move) and exploration_rate >= 0.2:
                if invalid_attempt_count >= 3 or random.random() < 0.25:
                    break
                invalid_attempt_count += 1
                continue  # try again
            break

        # for even better learning, test all possible moves and make the one that ends in a win (if possible)
        # this will both train the agent to make such a move or block it in the perspective of the opponent (see below)
        for test_move in board.get_available_columns():
            board.make_move(test_move)
            if board.get_winner() is not None:
                move = test_move
            board.undo_move()

        reward = -5  # default reward for generic move; discourages filling up the board to get small rewards
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
                        agent.process_feedback(from_state2, move, None, 100)
                    else:  # opponent would have blocked the win, which is way better than making a non-winning move
                        agent.process_feedback(from_state2, move, to_state2, 95)

        agent.process_feedback(from_state, move, None if game_over else to_state, reward)

        last_move = move
        total_reward[currently_playing_as] += reward

        if game_over:
            break

    avg_reward = (total_reward[PLAYER1]+total_reward[PLAYER2])/2
    return avg_reward, invalid_move


if __name__ == "__main__":
    run()

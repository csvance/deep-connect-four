import argparse
import datetime
import os
from enum import Enum, unique

import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


@unique
class C4Team(Enum):
    RED = 1
    BLACK = 2


@unique
class C4SlotState(Enum):
    EMPTY = 0
    RED = 1
    BLACK = 2


class C4TeamPerspectiveSlotState(Enum):
    EMPTY = 0
    SELF = 1
    ENEMY = 2


@unique
class C4Move(Enum):
    COLUMN1 = 0
    COLUMN2 = 1
    COLUMN3 = 2
    COLUMN4 = 3
    COLUMN5 = 4
    COLUMN6 = 5
    COLUMN7 = 6


@unique
class C4ActionResult(Enum):
    NONE = 0
    INVALID = 1
    VICTORY = 2
    TIE = 3


class ConnectFourGame(object):
    STATE_DIM = 6 * 7 * 3

    def __init__(self):
        self.turn = None
        self._state = None
        self._red_states = None
        self._red_labels = None
        self._black_states = None
        self._black_states = None
        self.winner = None
        self.reset()

    def reset(self):
        self.turn = 0
        self.winner = None
        self._state = np.zeros((6, 7))
        self._red_states = []
        self._red_labels = []
        self._black_states = []
        self._black_labels = []

    def _check_winner(self, row: int, column: int) -> C4ActionResult:

        team = self._state[row][column]

        # # Left / Right
        in_a_row = 0
        for item in self._state[row]:
            if item == team:
                in_a_row += 1
                if in_a_row == 4:
                    return C4ActionResult.VICTORY
            else:
                in_a_row = 0

        # Up / Down
        in_a_row = 0
        for item in self._state[:, column]:
            if item == team:
                in_a_row += 1
                if in_a_row == 4:
                    return C4ActionResult.VICTORY
            else:
                in_a_row = 0

        # Diagonal
        # Find border cell
        r = row
        c = column
        while True:
            if r < 0 or r >= 6 or c < 0 or c >= 7:
                r += 1
                c += 1
                break
            r -= 1
            c -= 1

        in_a_row = 0
        while True:
            if r < 0 or r >= 6 or c < 0 or c >= 7:
                break
            item = self._state[r][c]
            if item == team:
                in_a_row += 1
                if in_a_row == 4:
                    return C4ActionResult.VICTORY
            else:
                in_a_row = 0

            r += 1
            c += 1

        # Find border cell
        r = row
        c = column
        while True:
            if r < 0 or r >= 6 or c < 0 or c >= 7:
                r -= 1
                c += 1
                break
            r += 1
            c -= 1

        in_a_row = 0
        while True:
            if r < 0 or r >= 6 or c < 0 or c >= 7:
                break
            item = self._state[r][c]
            if item == team:
                in_a_row += 1
                if in_a_row == 4:
                    return C4ActionResult.VICTORY
            else:
                in_a_row = 0

            r -= 1
            c += 1

        return C4ActionResult.NONE

    def _apply_action(self, row: int, column: int, team: C4Team) -> C4ActionResult:

        if team == C4Team.RED:
            self._red_states.append(self.state(team))
            self._red_labels.append(C4Move(column).value)
        elif team == C4Team.BLACK:
            self._black_states.append(self.state(team))
            self._black_labels.append(C4Move(column).value)

        self._state[row][column] = team.value
        self.turn += 1
        return self._check_winner(row, column)

    def action(self, move: C4Move, team: C4Team) -> C4ActionResult:

        column = move.value

        # Put piece in first available row
        for row in reversed(range(0, 6)):
            if self._state[row][column] == C4SlotState.EMPTY.value:
                if self._apply_action(row, column, team) == C4ActionResult.VICTORY:
                    self.winner = team
                    return C4ActionResult.VICTORY
                return C4ActionResult.NONE

        if np.sum(self.valid_moves()) == 0:
            return C4ActionResult.TIE
        return C4ActionResult.INVALID

    def valid_moves(self) -> np.ndarray:
        valid_moves = []
        for column in range(0, 7):
            valid = False
            for row in reversed(range(0, 6)):
                if self._state[row][column] == C4SlotState.EMPTY.value:
                    valid = True
                    break
            if valid:
                valid_moves.append(1)
            else:
                valid_moves.append(0)

        return np.array(valid_moves)

    def state(self, perspective: C4Team) -> np.ndarray:

        state = self._state.copy()

        if perspective == C4Team.BLACK:
            np.place(state, state == C4Team.BLACK.value, [C4TeamPerspectiveSlotState.SELF.value])
            np.place(state, state == C4Team.RED.value, [C4TeamPerspectiveSlotState.ENEMY.value])
        elif perspective == C4Team.RED:
            np.place(state, state == C4Team.RED.value, [C4TeamPerspectiveSlotState.SELF.value])
            np.place(state, state == C4Team.BLACK.value, [C4TeamPerspectiveSlotState.ENEMY.value])

        def one_hot(idx: int, size: int):
            ret = [0.] * size
            ret[int(idx)] = 1.
            return ret

        one_hot_rows = np.zeros((6, 7, 3))
        for row_idx, row in enumerate(state):
            for column_idx, column in enumerate(row):
                one_hot_rows[row_idx][column_idx] = one_hot(column, 3)

        return one_hot_rows

    def display(self) -> str:

        output = "(1)(2)(3)(4)(5)(6)(7)\n"
        for row in self._state:
            for slot in row:
                if slot == C4SlotState.EMPTY.value:
                    output += "( )"
                elif slot == C4SlotState.BLACK.value:
                    output += "(Y)"
                elif slot == C4SlotState.RED.value:
                    output += "(R)"
            output += "\n"
        return output[:len(output) - 1]

    def current_turn(self):
        if self.turn % 2 == 0:
            return C4Team.RED
        else:
            return C4Team.BLACK

    def training_data(self):
        if self.winner is None:
            return None, None
        elif self.winner == C4Team.RED:
            return np.array(self._red_states), np.array(self._red_labels)
        elif self.winner == C4Team.BLACK:
            return np.array(self._black_states), np.array(self._black_labels)


class ConnectFourModel(object):
    def __init__(self, use_gpu=True, epsilon: float = 0., epsilon_decay: float = 0.995, epsilon_min=0.01):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        print("Initializing ConnectFourModel")
        print("epsilon: %f" % self.epsilon)
        print("epsilon_min: %f" % self.epsilon_min)
        print("epsilon_decay: %f" % self.epsilon_decay)

        model = Sequential()
        model.add(Dense(ConnectFourGame.STATE_DIM * 8, input_shape=(6, 7, 3), activation='relu'))
        model.add(Dense(ConnectFourGame.STATE_DIM * 8, activation='relu'))
        model.add(Flatten())
        model.add(Dense(len(C4Move), activation='softmax'))
        model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy')
        model.summary()
        self._model = model

        if use_gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            set_session(tf.Session(config=config))

    # Data is the game state, labels are the action taken
    def train(self, data, labels, reward=1):
        self._model.fit(data, labels, batch_size=1, verbose=0, epochs=reward)

    def predict(self, state, valid_moves: np.ndarray, epsilon_weight=1.0, argmax=False) -> C4Move:
        if np.random.rand() <= epsilon_weight * self.epsilon:
            potential_moves = []
            for idx in range(0, len(valid_moves)):
                if valid_moves[idx] == 1:
                    potential_moves.append(C4Move(idx))
            return np.random.choice(potential_moves)

        predictions = self._model.predict(np.array([state]))[0]

        # We only want valid moves
        predictions = predictions * valid_moves

        # Re-normalize
        sigma = np.sum(predictions)
        try:
            predictions = predictions / sigma
        except FloatingPointError:
            # If we had a floating point exception, it means no valid moves had non-zero p_values.
            # Choose a valid move at random
            potential_moves = []
            for idx in range(0, len(valid_moves)):
                if valid_moves[idx] == 1:
                    potential_moves.append(C4Move(idx))
            return np.random.choice(potential_moves)

        if argmax:
            return C4Move(np.argmax(predictions))
        else:
            return C4Move(np.random.choice([i for i in range(0, 7)], p=predictions))

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path='weights.h5'):
        self._model.save_weights(filepath=path)

    def load(self, path='weights.h5'):
        self._model.load_weights(filepath=path)


def human_vs_human():
    c4 = ConnectFourGame()

    while True:
        print(c4.display())

        current_turn = c4.current_turn()
        move = input("Move(%s): " % current_turn)
        if move == 'q':
            return
        move = int(move) - 1
        result = c4.action(C4Move(move), current_turn)
        print("Result: %s" % result)
        if result == C4ActionResult.VICTORY:
            c4.reset()
        elif result == C4ActionResult.TIE:
            c4.reset()


def human_vs_ai(weight_file):
    c4 = ConnectFourGame()
    c4ai = ConnectFourModel()
    try:
        if weight_file is not None:
            c4ai.load(weight_file)
    except OSError:
        print("Warning: could not load weights file!")
        pass
    print(c4.display())

    while True:

        current_team = c4.current_turn()
        valid_moves = c4.valid_moves()

        if current_team == C4Team.RED:
            move = input("Move(%s): " % current_team)
            if move == 'q':
                return
            move = int(move) - 1
            result = c4.action(C4Move(move), current_team)
        elif current_team == C4Team.BLACK:
            state = c4.state(current_team)
            move = c4ai.predict(state, valid_moves=valid_moves, argmax=True)
            result = c4.action(move, current_team)

        print(c4.display())

        print("Result: %s" % result)
        if result == C4ActionResult.VICTORY:
            c4.reset()
        elif result == C4ActionResult.TIE:
            c4.reset()


def ai_vs_ai(weights_file: str, epsilon: float, epsilon_decay: float, epsilon_min: float):
    time_string = datetime.datetime.now().strftime('%m-%d-%y-%H-%M-%S')
    game_no = 0
    red_wins = 0
    red_win_streak = 0
    red_loss_streak = 0
    black_wins = 0
    black_win_streak = 0
    black_loss_streak = 0

    c4 = ConnectFourGame()
    c4ai = ConnectFourModel(epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    try:
        if weights_file is not None:
            c4ai.load(weights_file)
    except OSError:
        print("Warning: could not load weights file!")
        pass

    while True:
        current_team = c4.current_turn()
        state = c4.state(current_team)

        valid_moves = c4.valid_moves()
        if np.sum(valid_moves) == 0:
            print("Tie!")

            # Reset
            c4.reset()
            continue

        epsilon_weight = 1.
        if current_team == C4Team.RED:
            epsilon_weight += red_loss_streak / 10.
        elif current_team == C4Team.BLACK:
            epsilon_weight += black_loss_streak / 10.
        move = c4ai.predict(state, valid_moves=valid_moves, epsilon_weight=epsilon_weight)

        result = c4.action(move, current_team)
        if result == C4ActionResult.VICTORY:
            # Update stats
            if current_team == C4Team.RED:
                red_wins += 1
                red_win_streak += 1
                red_loss_streak = 0
                black_win_streak = 0
            elif current_team == C4Team.BLACK:
                black_wins += 1
                black_win_streak += 1
                black_loss_streak = 0
                red_win_streak = 0

            print("Red: %d Black %d Epsilon: %f" % (red_wins, black_wins, c4ai.epsilon))
            print(c4.display())
            print("")

            # Train
            data, labels = c4.training_data()
            c4ai.train(data, labels)

            # Save Weights every 1000 games
            if game_no % 1000 == 0:
                c4ai.save('weights_' + str(game_no) + '_' + time_string + '.h5')

            game_no += 1

            # Reset
            c4.reset()

            # Decay epsilon
            c4ai.decay()


if __name__ == '__main__':
    np.seterr(all='raise')

    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('--weights-file')
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    parser.add_argument('--epsilon-min', type=float, default=0.01)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.mode == 'hvh':
        human_vs_human()
    elif args.mode == 'ava':
        ai_vs_ai(weights_file=args.weights_file, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay,
                 epsilon_min=args.epsilon_min)
    elif args.mode == 'hva':
        human_vs_ai(args.weights_file)
    else:
        print("Valid modes are: ")
        print("hvh - Human vs Human")
        print("hva - Huamn vs AI")
        print("ava - AI vs AI (Training)")

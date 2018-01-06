import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.layers import Dense, Flatten, Input, Conv2D, concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
from typing import List, Tuple
from c4_game import C4TeamPerspectiveSlotState, C4Move


class C4FeatureAnalyzer(object):
    def __init__(self, state: np.ndarray):
        self.state = state

    def analyze(self) -> Tuple[np.ndarray, np.ndarray]:
        moves = []
        for column in range(0, 7):
            moves.append(self.analyze_move(column))

        global_result = self.global_analysis()
        return np.array(global_result), np.array(moves)

    def global_analysis(self):

        normalized_column_counts = self.column_analysis()
        global_column_count = np.sum(normalized_column_counts) / 7.

        result = [global_column_count]
        return result

    def column_analysis(self) -> List:

        # Fill ratio for individual columns
        column_slot_filled_ratios = []
        for column in range(0, 7):
            piece_counts = np.bincount(self.state[:, column], minlength=3)
            column_slot_filled_ratios.append((piece_counts[C4TeamPerspectiveSlotState.SELF.value] +
                                              piece_counts[C4TeamPerspectiveSlotState.ENEMY.value]) / 6.)

        return column_slot_filled_ratios

    # TODO: verify ranges
    def analyze_vector(self, v: Tuple[np.ndarray, int]):

        vector = v[0]
        insertion_idx = v[1]

        # Can this create a viable connection?
        good_front = 0
        connect_self = 0
        for idx in range(min(len(vector), insertion_idx+1), min(len(vector), insertion_idx+4)):
            if vector[idx] == C4TeamPerspectiveSlotState.SELF.value:
                good_front += 1
                connect_self += 1
            elif vector[idx] == C4TeamPerspectiveSlotState.EMPTY.value:
                good_front += 1
            else:
                break

        friendly_in_a_row_front = 0
        for idx in range(min(len(vector), insertion_idx + 1), min(len(vector), insertion_idx + 4)):
            if vector[idx] == C4TeamPerspectiveSlotState.SELF.value:
                friendly_in_a_row_front += 1
            else:
                break

        # Now from the back
        good_back = 0
        for idx in reversed(range(max(0, insertion_idx-3), insertion_idx)):
            if vector[idx] == C4TeamPerspectiveSlotState.SELF.value:
                good_back += 1
                connect_self += 1
            elif vector[idx] == C4TeamPerspectiveSlotState.EMPTY.value:
                good_back += 1
            else:
                break

        friendly_in_a_row_back = 0
        for idx in reversed(range(max(0, insertion_idx - 3), insertion_idx)):
            if vector[idx] == C4TeamPerspectiveSlotState.SELF.value:
                friendly_in_a_row_back += 1
            else:
                break

        if good_front + good_back + 1 >= 4:
            potential_connection = 1.
        else:
            potential_connection = 0.

        block_front = 0
        block_enemy = 0
        # Will this block a viable connection for the enemy?
        for idx in range(min(len(vector), insertion_idx+1), min(len(vector), insertion_idx+4)):
            if vector[idx] == C4TeamPerspectiveSlotState.ENEMY.value:
                block_front += 1
                block_enemy += 1
            elif vector[idx] == C4TeamPerspectiveSlotState.EMPTY.value:
                block_front += 1
            else:
                break

        enemy_in_a_row_front = 0
        for idx in range(min(len(vector), insertion_idx + 1), min(len(vector), insertion_idx + 4)):
            if vector[idx] == C4TeamPerspectiveSlotState.ENEMY.value:
                enemy_in_a_row_front += 1
            else:
                break

        # Now from the back
        block_back = 0
        for idx in reversed(range(max(0, insertion_idx-3), insertion_idx)):
            if vector[idx] == C4TeamPerspectiveSlotState.ENEMY.value:
                block_back += 1
                block_enemy += 1
            elif vector[idx] == C4TeamPerspectiveSlotState.EMPTY.value:
                block_back += 1
            else:
                break

        enemy_in_a_row_back = 0
        for idx in reversed(range(max(0, insertion_idx - 3), insertion_idx)):
            if vector[idx] == C4TeamPerspectiveSlotState.ENEMY.value:
                enemy_in_a_row_back += 1
            else:
                break

        if block_front + block_back + 1 >= 4:
            potential_block = 1
        else:
            potential_block = 0

        return [potential_connection, potential_block, connect_self / 6., block_enemy / 6.,
                max(friendly_in_a_row_front, friendly_in_a_row_back) / 3.,
                max(enemy_in_a_row_front, enemy_in_a_row_back) / 3.]

    def analyze_move(self, column: int) -> List:

        # Find where the move would go
        last_empty = None
        for row in range(0, 6):
            if self.state[row][column] == C4TeamPerspectiveSlotState.EMPTY.value:
                last_empty = row

        if last_empty is None:
            return [0., 0., 0., 0., 0., 0.]

        # It is possible to make a move here, so lets analyze it
        row = last_empty

        vectors = []

        # Left / Right
        vectors.append((self.state[row], column))

        # Up Down
        vectors.append((self.state[:, column], row))

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

        vector = []
        insertion_idx = None
        while True:
            if r < 0 or r >= 6 or c < 0 or c >= 7:
                break
            vector.append(self.state[r][c])
            if r == row and c == column:
                insertion_idx = len(vector)
            r += 1
            c += 1

        vectors.append((np.array(vector), insertion_idx))

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

        vector = []
        insertion_idx = None
        while True:
            if r < 0 or r >= 6 or c < 0 or c >= 7:
                break
            vector.append(self.state[r][c])
            if r == row and c == column:
                insertion_idx = len(vector)

            r -= 1
            c += 1

        vectors.append((np.array(vector), insertion_idx))

        connect_space_normalized = 0
        block_space_normalized = 0
        connect_self_normalized = 0
        block_enemy_normalized = 0
        friendly_in_a_row_normalized = 0
        enemy_in_a_row_normalized = 0
        for v in vectors:
            f = self.analyze_vector(v)
            connect_space_normalized += f[0]
            block_space_normalized += f[1]
            connect_self_normalized += f[2]
            block_enemy_normalized += f[3]

            friendly_in_a_row_normalized = max(friendly_in_a_row_normalized, f[4])
            enemy_in_a_row_normalized = max(enemy_in_a_row_normalized, f[4])

        connect_space_normalized /= 4.
        block_space_normalized /= 4.
        connect_self_normalized /= 4.
        block_enemy_normalized /= 4.

        return [friendly_in_a_row_normalized, enemy_in_a_row_normalized]


class C4Model(object):
    def __init__(self, use_gpu=True, epsilon: float = 0., epsilon_decay: float = 0.9999, epsilon_min=0.05, gamma=0.95):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        model = Sequential()
        model.add(Conv2D(6 * 7, (4, 4), input_shape=(6, 7, 3), activation='elu'))
        model.add(Conv2D(6 * 7, (1, 2), input_shape=(6, 7, 3), activation='elu'))

        model.add(Flatten())
        model.add(Dense(len(C4Move), activation='linear'))
        model.compile(optimizer=Adam(lr=0.001), loss='mse')
        model.summary()
        self._model = model

        if use_gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            set_session(tf.Session(config=config))

    # Data is the game state, labels are the action taken
    def train(self, state_i: np.ndarray, action: C4Move, reward: float, state_f: np.ndarray, done: bool):

        if not done:
            target = reward + self.gamma * np.amax(self._model.predict(np.array([state_f]))[0])
        else:
            target = reward

        target_f = self._model.predict(np.array([state_i]))
        target_f[0][action] = target

        history = self._model.fit(np.array([state_i]), target_f, epochs=1, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return history

    def predict(self, state, valid_moves: np.ndarray) -> C4Move:
        if np.random.rand() <= self.epsilon:
            potential_moves = []
            for idx in range(0, len(valid_moves)):
                if valid_moves[idx] == 1:
                    potential_moves.append(C4Move(idx))
            return np.random.choice(potential_moves)

        predictions = self._model.predict(np.array([state]))[0]

        # We only want valid moves
        predictions = predictions * valid_moves

        return C4Move(np.argmax(predictions))

    def save(self, path='weights.h5'):
        self._model.save_weights(filepath=path)

    def load(self, path='weights.h5'):
        self._model.load_weights(filepath=path)

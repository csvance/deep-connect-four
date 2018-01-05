import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.layers import Dense, Flatten, Input, concatenate
from keras.models import Model
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
        result.extend(normalized_column_counts)
        return result

    def column_analysis(self) -> List:

        # Fill ratio for individual columns
        column_slot_filled_ratios = []
        for column in range(0, 7):
            piece_counts = np.bincount(self.state[:, column], minlength=3)
            column_slot_filled_ratios.append((piece_counts[C4TeamPerspectiveSlotState.SELF.value] +
                                              piece_counts[C4TeamPerspectiveSlotState.ENEMY.value]) / 6.)

        return column_slot_filled_ratios

    def analyze_vector(self, v: Tuple[np.ndarray, int]):

        vector = v[0]
        insertion_idx = v[1]

        # Can this create a viable connection?
        good_front = 0
        for idx in range(insertion_idx+1, len(vector)):
            if vector[idx] == C4TeamPerspectiveSlotState.SELF.value or vector[idx] == C4TeamPerspectiveSlotState.EMPTY.value:
                good_front += 1
            else:
                break

        # Now from the back
        good_back = 0
        for idx in reversed(range(0, insertion_idx)):
            if vector[idx] == C4TeamPerspectiveSlotState.SELF.value or vector[idx] == C4TeamPerspectiveSlotState.EMPTY.value:
                good_back += 1
            else:
                break

        if good_front + good_back + 1 >= 4:
            potential_connection = 1.
        else:
            potential_connection = 0.

        block_front = 0

        # Will this block a viable connection for the enemy?
        for idx in range(insertion_idx+1, len(vector)):
            if vector[idx] == C4TeamPerspectiveSlotState.ENEMY.value or vector[idx] == C4TeamPerspectiveSlotState.EMPTY.value:
                block_front += 1
            else:
                break

        # Now from the back
        block_back = 0
        for idx in reversed(range(0, insertion_idx)):
            if vector[idx] == C4TeamPerspectiveSlotState.ENEMY.value or vector[idx] == C4TeamPerspectiveSlotState.EMPTY.value:
                block_back += 1
            else:
                break

        if block_front + block_back + 1 >= 4:
            potential_block = 1
        else:
            potential_block = 0

        return [potential_connection, potential_block]

    def analyze_move(self, column: int) -> List:

        # Find where the move would go
        last_empty = None
        for row in range(0, 6):
            if self.state[row][column] == C4TeamPerspectiveSlotState.EMPTY.value:
                last_empty = row

        if last_empty is None:
            return [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]

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

        features = []
        for v in vectors:
            features.append(self.analyze_vector(v))
        return features


class C4Model(object):
    def __init__(self, use_gpu=True, epsilon: float = 0., epsilon_decay: float = 0.995, epsilon_min=0.01):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        global_input = Input(shape=(8,), name='globals')
        g = Dense(8, activation='relu')(global_input)

        move_input = Input(shape=(7, 4, 2), name='moves')
        m = Dense(64, activation='relu')(move_input)
        m = Flatten()(m)

        c = concatenate([g, m])
        c = Dense(32, activation='relu')(c)

        output = Dense(len(C4Move), activation='softmax')(c)

        model = Model(inputs=[global_input, move_input], outputs=[output])
        model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy')
        model.summary()
        self._model = model

        if use_gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            set_session(tf.Session(config=config))

    # Data is the game state, labels are the action taken
    def train(self, data, labels, reward=1):
        return self._model.fit(data, labels, batch_size=1, verbose=1, epochs=reward)

    def predict(self, state, valid_moves: np.ndarray, epsilon_weight=1.0, argmax=False) -> C4Move:
        if np.random.rand() <= epsilon_weight * self.epsilon:
            potential_moves = []
            for idx in range(0, len(valid_moves)):
                if valid_moves[idx] == 1:
                    potential_moves.append(C4Move(idx))
            return np.random.choice(potential_moves)

        predictions = self._model.predict(state)[0]

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

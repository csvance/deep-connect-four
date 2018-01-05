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

    def analyze(self) -> Tuple[np.ndarray,np.ndarray]:
        moves = []
        for column in range(0, 7):
            moves.append(self.analyze_move(column))

        result = self.global_analysis()
        return np.array(result), np.array(moves)

    def global_analysis(self):

        normalized_column_counts = self.current_column_analysis()
        global_column_count = np.sum(normalized_column_counts) / 7.

        result = [global_column_count]
        result.extend(normalized_column_counts)
        return result

    def current_column_analysis(self) -> List:

        # Fill ratio for individual columns
        column_slot_filled_ratios = []
        for column in range(0, 7):
            piece_counts = np.bincount(self.state[:, column], minlength=3)
            column_slot_filled_ratios.append((piece_counts[C4TeamPerspectiveSlotState.SELF.value] +
                                              piece_counts[C4TeamPerspectiveSlotState.ENEMY.value]) / 6.)

        return column_slot_filled_ratios

    def analyze_move_pre_post(self, pre_row: np.ndarray, post_row: np.ndarray):
        assert len(pre_row) == len(post_row)

        def stats(empty, me, enemy):
            def _stat(l):
                _min = min(l) / 4.
                _max = max(l) / 4.
                _sum = sum(l) / len(pre_row)
                _avg = _sum / len(l)
                return [_min, _max, _sum, _avg]

            return np.array([_stat(empty), _stat(me), _stat(enemy)])

        def scan(row: np.ndarray):

            self_in_a_rows = []
            self_in_a_row = 0

            enemy_in_a_rows = []
            enemy_in_a_row = 0

            empty_in_a_rows = []
            empty_in_a_row = 0

            for item in row:
                if item == C4TeamPerspectiveSlotState.SELF.value:

                    self_in_a_row += 1

                    if empty_in_a_row > 0:
                        empty_in_a_rows.append(empty_in_a_row)
                        empty_in_a_row = 0
                    elif enemy_in_a_row > 0:
                        enemy_in_a_rows.append(enemy_in_a_row)
                        enemy_in_a_row = 0

                elif item == C4TeamPerspectiveSlotState.ENEMY.value:
                    enemy_in_a_row += 1

                    if self_in_a_row > 0:
                        self_in_a_rows.append(self_in_a_row)
                        self_in_a_row = 0
                    elif empty_in_a_row > 0:
                        empty_in_a_rows.append(empty_in_a_row)
                        empty_in_a_row = 0

                elif item == C4TeamPerspectiveSlotState.EMPTY.value:
                    empty_in_a_row += 1

                    if self_in_a_row > 0:
                        self_in_a_rows.append(self_in_a_row)
                        self_in_a_row = 0
                    elif enemy_in_a_row > 0:
                        enemy_in_a_rows.append(enemy_in_a_row)
                        enemy_in_a_row = 0

            if empty_in_a_row:
                empty_in_a_rows.append(empty_in_a_row)
            if len(empty_in_a_rows) == 0:
                empty_in_a_rows.append(0)

            if self_in_a_row:
                self_in_a_rows.append(self_in_a_row)
            if len(self_in_a_rows) == 0:
                self_in_a_rows.append(0)

            if enemy_in_a_row:
                enemy_in_a_rows.append(enemy_in_a_row)
            if len(enemy_in_a_rows) == 0:
                enemy_in_a_rows.append(0)

            return empty_in_a_rows, self_in_a_rows, enemy_in_a_rows

        pre_empty_in_a_rows, pre_self_in_a_rows, pre_enemy_in_a_rows = scan(pre_row)
        pre_stats = stats(pre_empty_in_a_rows, pre_self_in_a_rows, pre_enemy_in_a_rows)

        post_empty_in_a_rows, post_self_in_a_rows, post_enemy_in_a_rows = scan(post_row)
        post_stats = stats(post_empty_in_a_rows, post_self_in_a_rows, post_enemy_in_a_rows)

        return pre_stats, post_stats

    def analyze_move(self, column: int) -> np.ndarray:

        # Find where the move would go
        last_empty = None
        for row in range(0, 6):
            if self.state[row][column] == C4TeamPerspectiveSlotState.EMPTY.value:
                last_empty = row
            else:
                return np.zeros((4, 3, 4))

        # It is possible to make a move here, so lets analyze it
        row = last_empty

        # Left / Right
        pre_row = self.state[row]
        post_row = pre_row.copy()
        post_row[column] = C4TeamPerspectiveSlotState.SELF.value
        pre_left_right_stats, post_left_right_stats = self.analyze_move_pre_post(pre_row, post_row)
        left_right_stats = post_left_right_stats - pre_left_right_stats

        # Up Down
        pre_row = self.state[:, column]
        post_row = pre_row.copy()
        post_row[row] = C4TeamPerspectiveSlotState.SELF.value
        pre_up_down_stats, post_up_down_stats = self.analyze_move_pre_post(pre_row, post_row)

        up_down_stats = post_up_down_stats - pre_up_down_stats

        # Look what the enemy might do playing ontop of us
        if row <= 4:
            enemy_row = post_row.copy()
            enemy_row[row + 1] = C4TeamPerspectiveSlotState.ENEMY.value
            _, enemy_up_down_stats = self.analyze_move_pre_post(post_row, enemy_row)

            up_down_stats = (up_down_stats + (enemy_up_down_stats - post_up_down_stats)) / 2.

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

        pre_row = []
        post_row = []
        while True:
            if r < 0 or r >= 6 or c < 0 or c >= 7:
                break
            pre_row.append(self.state[r][c])
            if c == column and r == row:
                post_row.append(C4TeamPerspectiveSlotState.SELF.value)
            else:
                post_row.append(self.state[r][c])

            r += 1
            c += 1

        pre_diag_stats_pp, post_diag_stats_pp = self.analyze_move_pre_post(np.array(pre_row), np.array(post_row))
        diag_stats_pp = post_diag_stats_pp - pre_diag_stats_pp

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

        pre_row = []
        post_row = []
        while True:
            if r < 0 or r >= 6 or c < 0 or c >= 7:
                break
            pre_row.append(self.state[r][c])
            if c == column and r == row:
                post_row.append(C4TeamPerspectiveSlotState.SELF.value)
            else:
                post_row.append(self.state[r][c])

            r -= 1
            c += 1

        pre_diag_stats_mp, post_diag_stats_mp = self.analyze_move_pre_post(np.array(pre_row), np.array(post_row))
        diag_stats_mp = post_diag_stats_mp - pre_diag_stats_mp

        return np.array([up_down_stats, left_right_stats, diag_stats_pp, diag_stats_mp])


class C4Model(object):
    def __init__(self, use_gpu=True, epsilon: float = 0., epsilon_decay: float = 0.995, epsilon_min=0.01):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        global_input = Input(shape=(8,), name='globals')
        g = Dense(8, activation='relu')(global_input)

        move_input = Input(shape=(7, 4, 3, 4), name='moves')
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
        return self._model.fit(data, labels, batch_size=1, verbose=0, epochs=reward)

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

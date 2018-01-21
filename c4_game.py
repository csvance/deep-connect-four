import pickle
import random
from collections import deque
from enum import Enum, unique

import numpy as np


@unique
class C4Team(Enum):
    RED = 1
    BLACK = 2


class C4SlotState(Enum):
    EMPTY = 0
    SELF = 1
    ENEMY = 2


@unique
class C4Action(Enum):
    COLUMN1 = 0
    COLUMN2 = 1
    COLUMN3 = 2
    COLUMN4 = 3
    COLUMN5 = 4
    COLUMN6 = 5
    COLUMN7 = 6


@unique
class C4MoveResult(Enum):
    NONE = 0
    INVALID = 1
    VICTORY = 2
    TIE = 3


class C4Experience(object):
    def __init__(self, action: C4Action, result: C4MoveResult, old_state: 'C4State', new_state: 'C4State',
                 reward: float, terminal: bool = False):
        self.result = result
        self.old_state = old_state
        self.new_state = new_state
        self.reward = reward
        self.terminal = terminal
        self.action = action


class C4State(object):
    def __init__(self, state: np.ndarray = None):
        if state is None:
            self.state = np.zeros((6, 7), dtype=np.int8)
        else:
            self.state = state

    def __iter__(self):
        for row in self.state:
            yield row

    def __cmp__(self, other):
        return np.array_equal(self.state, other.state)

    def copy(self):
        return C4State(state=self.state.copy())

    def move(self, move: C4Action) -> C4MoveResult:

        column = move.value

        # Put piece in first available row
        for row in reversed(range(0, 6)):
            if self.state[row][column] == C4SlotState.EMPTY.value:
                self.state[row][column] = C4SlotState.SELF.value

                if self._check_winner(row, column):
                    return C4MoveResult.VICTORY

                if np.sum(self.valid_moves()) == 0:
                    return C4MoveResult.TIE

                return C4MoveResult.NONE

        if np.sum(self.valid_moves()) == 0:
            return C4MoveResult.TIE

        return C4MoveResult.INVALID

    def valid_moves(self) -> np.ndarray:
        valid_moves = []
        for column in range(0, 7):
            valid = False
            for row in reversed(range(0, 6)):
                if self.state[row][column] == C4SlotState.EMPTY.value:
                    valid = True
                    break
            if valid:
                valid_moves.append(1)
            else:
                valid_moves.append(0)

        return np.array(valid_moves)

    def _check_winner(self, row: int, column: int) -> bool:

        team = self.state[row][column]

        # # Left / Right
        in_a_row = 0
        for item in self.state[row]:
            if item == team:
                in_a_row += 1
                if in_a_row == 4:
                    return True
            else:
                in_a_row = 0

        # Up / Down
        in_a_row = 0
        for item in self.state[:, column]:
            if item == team:
                in_a_row += 1
                if in_a_row == 4:
                    return True
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
            item = self.state[r][c]
            if item == team:
                in_a_row += 1
                if in_a_row == 4:
                    return True
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
            item = self.state[r][c]
            if item == team:
                in_a_row += 1
                if in_a_row == 4:
                    return True
            else:
                in_a_row = 0

            r -= 1
            c += 1

        return False

    def invert_perspective(self):
        np.place(self.state, self.state == C4SlotState.SELF.value, [127])
        np.place(self.state, self.state == C4SlotState.ENEMY.value, [C4SlotState.SELF.value])
        np.place(self.state, self.state == 127, [C4SlotState.ENEMY.value])

    def move_values(self) -> np.ndarray:

        def move_value(vector):
            self_in_a_row = 0.
            self_seperated = 0.
            enemy_in_a_row = 0.
            enemy_seperated = 0.

            r_end = False
            for idx, i in enumerate(vector):
                if i == C4SlotState.ENEMY.value:
                    if not r_end:
                        enemy_in_a_row += 1.
                    else:
                        enemy_seperated += 1.
                elif i == C4SlotState.SELF.value:
                    break
                else:
                    r_end = True

            r_end = False
            for idx, i in enumerate(vector):
                if i == C4SlotState.SELF.value:
                    if not r_end:
                        self_in_a_row += 1.
                    else:
                        self_seperated += 1.
                elif i == C4SlotState.ENEMY.value:
                    break
                else:
                    r_end = True

            return self_in_a_row, self_seperated, enemy_in_a_row, enemy_seperated

        def valid_index(row: int, col: int) -> bool:
            if col < 0 or col > 6:
                return False
            if row < 0 or row > 5:
                return False
            return True

        height = np.array(self.column_height(), dtype=np.int8)
        height = 5 - height

        self_values = []
        enemy_values = []

        for col_index in range(0, 7):
            # Left
            v = self.state[height[col_index]][max(0, col_index - 3):col_index][::-1]
            ss_left, s_left, ee_left, e_left = move_value(v)

            # Right
            v = self.state[height[col_index]][col_index + 1:col_index + 4]
            ss_right, s_right, ee_right, e_right = move_value(v)

            # Down
            v = self.state[:, col_index][height[col_index] + 1:height[col_index] + 4]
            ss_down, s_down, ee_down, e_down = move_value(v)

            # Up Right
            start_row = height[col_index]
            v = []
            for i in range(1, 4):
                if not valid_index(col_index + i, start_row + i):
                    break
                v.append(self.state[col_index + i][start_row + i])
            ss_up_right, s_up_right, ee_up_right, e_up_right = move_value(v)

            # Down Right
            start_row = height[col_index]
            v = []
            for i in range(1, 4):
                if not valid_index(start_row - i, col_index + i):
                    break
                v.append(self.state[start_row - i][col_index + i])
            ss_down_right, s_down_right, ee_down_right, e_down_right = move_value(v)

            # Up Left
            start_row = height[col_index]
            v = []
            for i in range(1, 4):
                if not valid_index(start_row + i, col_index - i):
                    break
                v.append(self.state[start_row + i][col_index - i])
            ss_up_left, s_up_left, ee_up_left, e_up_left = move_value(v)

            # Down Left
            start_row = height[col_index]
            v = []
            for i in range(1, 4):
                if not valid_index(start_row - i, col_index - i):
                    break
                v.append(self.state[start_row - i][col_index - i])
            ss_down_left, s_down_left, ee_down_left, e_down_left = move_value(v)

            ss = [min(3., ss_down), min(3., ss_right + ss_left), min(3., ss_up_right + ss_down_left),
                  min(3., ss_down_right + ss_up_left)]
            ee = [min(3., ee_down), min(3., ee_right + ee_left), min(3., ee_up_right + ee_down_left),
                  min(3., ee_down_right + ee_up_left)]

            s = [min(3., s_down), min(3., s_right + s_left), min(3., s_up_right + s_down_left),
                 min(3., s_down_right + s_up_left)]
            e = [min(3., e_down), min(3., e_right + e_left), min(3., e_up_right + e_down_left),
                 min(3., e_down_right + e_up_left)]

            self_values.append(ss + s)
            enemy_values.append(ee + e)

        ret_list = np.array([self_values, enemy_values])
        ret_list = ret_list.swapaxes(0, 2)
        ret_list = ret_list.swapaxes(0, 1)

        return np.array([ret_list]) / 3.

    def one_hot(self) -> np.ndarray:
        ret_state = np.zeros((6, 7, 2), dtype=np.int8)
        for row_idx, row in enumerate(self.state):
            for col_idx, col in enumerate(row):
                if col == C4SlotState.EMPTY.value:
                    pass
                elif col == C4SlotState.SELF.value:
                    ret_state[row_idx][col_idx][0] = 1
                elif col == C4SlotState.ENEMY.value:
                    ret_state[row_idx][col_idx][1] = 1
        return np.array([ret_state])

    def column_height(self) -> list:
        heights = []
        for column_idx in range(0, 7):
            vector = self.state[:, column_idx]
            height = 0
            for v in reversed(vector):
                if v != C4SlotState.EMPTY.value:
                    height += 1
                else:
                    break
            heights.append(height)

        return heights

    def state_representation(self):
        return [self.one_hot(), self.move_values(), np.array([self.column_height()]) / 5.]


class C4Game(object):

    def __init__(self):
        self.turn = 0
        self.state = C4State()
        self.winner = None

        self.red_memories = []
        self.black_memories = []

        self.last_victory = None
        self.duplicate = False

        self.first_turn = C4Team.RED

        self.memories = deque(maxlen=50000)

        self.red_score = 0.
        self.black_score = 0.

        self._red_last_score = 0.
        self._black_last_score = 0.

    def reset(self):
        self.turn = 0
        self.state = C4State()
        self.winner = None
        self.duplicate = False

        if self.first_turn == C4Team.RED:
            self.first_turn = C4Team.BLACK
        else:
            self.first_turn = C4Team.RED

        self.red_memories = []
        self.black_memories = []

        self.red_score = 0.
        self.black_score = 0.

        self._red_last_score = 0.
        self._black_last_score = 0.

    def action(self, action: C4Action) -> C4MoveResult:

        old_state = self.state.copy()
        move_result = self.state.move(action)
        new_state = self.state.copy()

        reward = None
        done = None
        if move_result == C4MoveResult.VICTORY:
            if self.last_victory is not None:
                if np.array_equal(self.last_victory.state, self.state.state):
                    self.duplicate = True
                self.last_victory.invert_perspective()

            self.last_victory = self.state.copy()
            reward = 1.
            done = True
        elif move_result == C4MoveResult.TIE:
            reward = 0.
            done = True
        elif move_result == C4MoveResult.INVALID:
            raise ValueError
        elif move_result == C4MoveResult.NONE:
            reward = 0.
            done = False

        experience = C4Experience(action=action, result=move_result,
                                  old_state=old_state, new_state=new_state,
                                  reward=reward, terminal=done)

        if not self.duplicate:
            if self.current_turn() == C4Team.RED:
                self.red_memories.append(experience)

                if move_result == C4MoveResult.VICTORY:
                    self.black_memories[-1].reward = -1.
                    self.black_memories[-1].terminal = True

                    self.memories.extend(self.red_memories)
                    self.memories.extend(self.black_memories)

            elif self.current_turn() == C4Team.BLACK:
                self.black_memories.append(experience)

                if move_result == C4MoveResult.VICTORY:
                    self.red_memories[-1].reward = -1.
                    self.red_memories[-1].terminal = True

                    self.memories.extend(self.red_memories)
                    self.memories.extend(self.black_memories)

        if move_result == C4MoveResult.NONE:
            self.turn += 1
            self.state.invert_perspective()

        return move_result

    def best_action(self, valid_moves: list, epsilon: float = 0.):

        if np.random.rand() <= epsilon:
            potential_moves = []
            for idx in range(0, len(valid_moves)):
                if valid_moves[idx] == 1:
                    potential_moves.append(C4Action(idx))
            return np.random.choice(potential_moves)

        self_scores = []
        enemy_scores = []

        move_values = self.state.move_values()[0]
        for move_idx, move in enumerate(valid_moves):
            if move == 0:
                move_values[move_idx] = -9999.

        for action_idx, action in enumerate(move_values):
            self_max_index = action_idx
            self_max_value = np.amax(action[:4, 0])
            enemy_max_index = action_idx
            enemy_max_value = np.amax(action[:4, 1])

            self_scores.append([self_max_index, self_max_value])
            enemy_scores.append([enemy_max_index, enemy_max_value])

        self_scores.sort(key=lambda x: x[1], reverse=True)
        enemy_scores.sort(key=lambda x: x[1], reverse=True)

        if self_scores[0][1] >= enemy_scores[0][1]:
            alike_scores = []
            for score in self_scores:
                if score[1] == self_scores[0][1]:
                    alike_scores.append(score[0])

            choice = np.random.choice(alike_scores)
            return C4Action(choice)
        else:
            alike_scores = []
            for score in enemy_scores:
                if score[1] == enemy_scores[0][1]:
                    alike_scores.append(score[0])

            choice = np.random.choice(alike_scores)
            return C4Action(choice)

    def display(self) -> str:

        turn = self.current_turn()

        output = "(1)(2)(3)(4)(5)(6)(7)\n"
        for row in self.state:
            for slot in row:
                if slot == C4SlotState.EMPTY.value:
                    output += "( )"
                if turn == C4Team.RED:
                    if slot == C4SlotState.SELF.value:
                        output += "(R)"
                    elif slot == C4SlotState.ENEMY.value:
                        output += "(B)"
                elif turn == C4Team.BLACK:
                    if slot == C4SlotState.SELF.value:
                        output += "(B)"
                    elif slot == C4SlotState.ENEMY.value:
                        output += "(R)"
            output += "\n"
        return output[:len(output) - 1]

    def current_turn(self):
        if self.first_turn == C4Team.RED:
            if self.turn % 2 == 0:
                return C4Team.RED
            else:
                return C4Team.BLACK
        else:
            if self.turn % 2 == 0:
                return C4Team.BLACK
            else:
                return C4Team.RED

    def sample(self, batch_size=32):

        if len(self.memories) < batch_size:
            return None

        # Shuffle
        return random.sample(self.memories, batch_size)

    def save(self):
        pickle.dump(self.memories, open('memories.p', 'wb'))

    def load(self):
        self.memories = pickle.load(open('memories.p', 'rb'))

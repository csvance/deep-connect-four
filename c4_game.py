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


class C4ActionResult(object):
    def __init__(self, action: C4Action, result: C4MoveResult, old_state: 'C4State', new_state: 'C4State',
                 reward: float, done: bool = False):
        self.result = result
        self.old_state = old_state
        self.new_state = new_state
        self.reward = reward
        self.done = done
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
        return

    def one_hot(self) -> np.ndarray:
        one_hot_state = np.zeros((6, 7, 2), dtype=np.int8)
        for row_idx, row in enumerate(self.state):
            for col_idx, column in enumerate(row):
                if column == C4SlotState.SELF.value:
                    one_hot_state[row_idx][col_idx][0] = 1
                elif column == C4SlotState.ENEMY.value:
                    one_hot_state[row_idx][col_idx][1] = 1

        return one_hot_state

    def normalized(self):
        ret_state = self.state.copy()
        ret_state = ret_state.reshape((6, 7, 1))
        return ret_state / 2.


class C4Game(object):

    def __init__(self):
        self.turn = 0
        self.state = C4State()
        self.winner = None

        self.red_memories = []
        self.black_memories = []

        self.last_victory = None
        self.duplicate = False

        self.normal_memories = deque(maxlen=10000)
        self.win_loss_memories = deque(maxlen=200)

    def reset(self):
        self.turn = 0
        self.state = C4State()
        self.winner = None
        self.duplicate = False

        self.red_memories = []
        self.black_memories = []

    def action(self, action: C4Action) -> C4MoveResult:

        old_state = self.state.copy()
        move_result = self.state.move(action)
        new_state = self.state

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
            reward = 0.
            done = False
        elif move_result == C4MoveResult.NONE:
            reward = 0.1
            done = False

        action_result = C4ActionResult(action=action, result=move_result, old_state=old_state, new_state=new_state,
                                       reward=reward, done=done)

        if not self.duplicate:
            if self.current_turn() == C4Team.RED:
                self.red_memories.append(action_result)

                if move_result == C4MoveResult.VICTORY:
                    self.black_memories[-1].reward = -1.
                    self.black_memories[-1].done = True

                    self.win_loss_memories.append(self.red_memories[-1])
                    self.win_loss_memories.append(self.black_memories[-1])

                    self.normal_memories.extend(self.red_memories[0:-1])
                    self.normal_memories.extend(self.black_memories[0:-1])

            elif self.current_turn() == C4Team.BLACK:
                self.black_memories.append(action_result)

                if move_result == C4MoveResult.VICTORY:
                    self.red_memories[-1].reward = -1.
                    self.red_memories[-1].done = True

                    self.win_loss_memories.append(self.red_memories[-1])
                    self.win_loss_memories.append(self.black_memories[-1])

                    self.normal_memories.extend(self.red_memories[0:-1])
                    self.normal_memories.extend(self.black_memories[0:-1])

        if move_result == C4MoveResult.NONE:
            self.turn += 1
            self.state.invert_perspective()

        return move_result

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
        if self.turn % 2 == 0:
            return C4Team.RED
        else:
            return C4Team.BLACK

    def sample(self, batch_size=21, win_loss_samples=2):

        if len(self.normal_memories) < batch_size:
            return None
        if len(self.win_loss_memories) < win_loss_samples:
            return None

        win_loss_batch_size = win_loss_samples
        normal_batch_size = batch_size - win_loss_batch_size

        normal_memories = random.sample(self.normal_memories, normal_batch_size)
        win_loss_memories = random.sample(self.win_loss_memories, win_loss_batch_size)

        combined_batch = []
        combined_batch.extend(normal_memories)
        combined_batch.extend(win_loss_memories)

        # Shuffle
        return random.sample(combined_batch, len(combined_batch))

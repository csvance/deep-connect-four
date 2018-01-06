import random
from collections import deque
from enum import Enum, unique

import numpy as np


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


class C4Game(object):

    def __init__(self):
        self.turn = None
        self.state = None
        self.winner = None
        self.red_memory = None
        self.black_memory = None

        self.winning_memories = deque(maxlen=2000)
        self.losing_memories = deque(maxlen=2000)
        self.win_step_memories = deque(maxlen=10)
        self.loss_step_memories = deque(maxlen=10)

        self.reset()

    def reset(self):
        self.turn = 0
        self.state = np.zeros((6, 7), dtype=np.int8)
        self.winner = None

        self.red_memory = []
        self.black_memory = []

    def _check_winner(self, row: int, column: int) -> C4ActionResult:

        team = self.state[row][column]

        # # Left / Right
        in_a_row = 0
        for item in self.state[row]:
            if item == team:
                in_a_row += 1
                if in_a_row == 4:
                    return C4ActionResult.VICTORY
            else:
                in_a_row = 0

        # Up / Down
        in_a_row = 0
        for item in self.state[:, column]:
            if item == team:
                in_a_row += 1
                if in_a_row == 4:
                    return C4ActionResult.VICTORY
            else:
                in_a_row = 0

        # TODO: Add diagnol win condition back into game
        return C4ActionResult.NONE

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
            item = self.state[r][c]
            if item == team:
                in_a_row += 1
                if in_a_row == 4:
                    return C4ActionResult.VICTORY
            else:
                in_a_row = 0

            r -= 1
            c += 1

        return C4ActionResult.NONE

    def reward(self):
        if self.turn < 3:
            return (self.turn + 1) / 4.
        return 1 - (self.turn / 42.)

    def _apply_action(self, row: int, column: int, team: C4Team) -> C4ActionResult:

        old_state = self.perspective_state(team)
        self.state[row][column] = team.value
        self.turn += 1
        new_state = self.perspective_state(team)

        action_result = self._check_winner(row, column)

        reward = 0.

        if team == C4Team.RED:
            self.red_memory.append((old_state, C4Move(column).value, reward, new_state))
        elif team == C4Team.BLACK:
            self.black_memory.append((old_state, C4Move(column).value, reward, new_state))

        # If someone won store their data
        if action_result == C4ActionResult.VICTORY:
            if team == C4Team.RED:

                # Replay Winner
                for item_idx, item in enumerate(self.red_memory):
                    if item_idx == len(self.red_memory) - 1:
                        self.win_step_memories.append((item[0], item[1], 1., item[3], True))
                    else:
                        self.winning_memories.append((item[0], item[1], 0., item[3], False))

                # Replay Loser
                for item_idx, item in enumerate(self.black_memory):
                    if item_idx == len(self.black_memory) - 1:
                        self.loss_step_memories.append((item[0], item[1], -1., item[3], True))
                    else:
                        self.losing_memories.append((item[0], item[1], 0., item[3], False))

            elif team == C4Team.BLACK:

                # Replay Winner
                for item_idx, item in enumerate(self.black_memory):
                    if item_idx == len(self.black_memory) - 1:
                        self.win_step_memories.append((item[0], item[1], 1., item[3], True))
                    else:
                        self.winning_memories.append((item[0], item[1], 0., item[3], False))

                # Replay Loser
                for item_idx, item in enumerate(self.red_memory):
                    if item_idx == len(self.red_memory) - 1:
                        self.loss_step_memories.append((item[0], item[1], -1., item[3], True))
                    else:
                        self.losing_memories.append((item[0], item[1], 0., item[3], False))

        return action_result

    def action(self, move: C4Move, team: C4Team) -> C4ActionResult:

        column = move.value

        # Put piece in first available row
        for row in reversed(range(0, 6)):
            if self.state[row][column] == C4SlotState.EMPTY.value:
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
                if self.state[row][column] == C4SlotState.EMPTY.value:
                    valid = True
                    break
            if valid:
                valid_moves.append(1)
            else:
                valid_moves.append(0)

        return np.array(valid_moves)

    def perspective_state(self, perspective: C4Team) -> np.ndarray:

        state = self.state.copy()

        # TODO: measure and optimize this
        if perspective == C4Team.BLACK:
            np.place(state, state == C4Team.BLACK.value, [127])
            np.place(state, state == C4Team.RED.value, [-1])
            np.place(state, state == 127, [1])
        elif perspective == C4Team.RED:
            np.place(state, state == C4Team.RED.value, [127])
            np.place(state, state == C4Team.BLACK.value, [-1])
            np.place(state, state == 127, [1])

        dense_state = state.reshape((6, 7, 1))

        return dense_state

    def display(self) -> str:

        output = "(1)(2)(3)(4)(5)(6)(7)\n"
        for row in self.state:
            for slot in row:
                if slot == C4SlotState.EMPTY.value:
                    output += "( )"
                elif slot == C4SlotState.BLACK.value:
                    output += "(B)"
                elif slot == C4SlotState.RED.value:
                    output += "(R)"
            output += "\n"
        return output[:len(output) - 1]

    def current_turn(self):
        if self.turn % 2 == 0:
            return C4Team.RED
        else:
            return C4Team.BLACK

    def training_data(self, batch_size=128, wins: int = 2, losses: int = 2):
        bulk_size = batch_size - (wins + losses)
        win_size = int(bulk_size / 2)
        loss_size = int(bulk_size / 2)

        while win_size + loss_size + wins + losses < batch_size:
            win_size += 1

        if win_size > len(self.winning_memories) or loss_size > len(self.losing_memories):
            return None

        win_samples = random.sample(self.winning_memories, win_size)
        win_step_samples = random.sample(self.win_step_memories, wins)

        loss_samples = random.sample(self.losing_memories, loss_size)
        loss_step_samples = random.sample(self.loss_step_memories, losses)

        batch = []
        batch.extend(win_samples)
        batch.extend(loss_samples)
        batch.extend(win_step_samples)
        batch.extend(loss_step_samples)

        return random.sample(batch, batch_size)

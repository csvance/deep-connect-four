from enum import Enum, unique
import numpy as np
from typing import Tuple, Optional


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
        self._state = np.zeros((6, 7), dtype=np.int8)
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

        state = self.state(team)

        if team == C4Team.RED:
            self._red_states.append(state)
            self._red_labels.append(C4Move(column).value)
        elif team == C4Team.BLACK:
            self._black_states.append(state)
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

        # TODO: measure and optimize this
        if perspective == C4Team.BLACK:
            np.place(state, state == C4Team.BLACK.value, [C4TeamPerspectiveSlotState.SELF.value+4])
            np.place(state, state == C4Team.RED.value, [C4TeamPerspectiveSlotState.ENEMY.value])
            np.place(state, state == C4TeamPerspectiveSlotState.SELF.value+4, [C4TeamPerspectiveSlotState.SELF.value])
        elif perspective == C4Team.RED:
            np.place(state, state == C4Team.RED.value, [C4TeamPerspectiveSlotState.SELF.value+4])
            np.place(state, state == C4Team.BLACK.value, [C4TeamPerspectiveSlotState.ENEMY.value])
            np.place(state, state == C4TeamPerspectiveSlotState.SELF.value+4, [C4TeamPerspectiveSlotState.SELF.value])

        return state

    def display(self) -> str:

        output = "(1)(2)(3)(4)(5)(6)(7)\n"
        for row in self._state:
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

    def training_data(self) -> Optional[Tuple]:
        if self.winner is None:
            return None
        elif self.winner == C4Team.RED:
            return self._red_states, self._red_labels
        elif self.winner == C4Team.BLACK:
            return self._black_states, self._black_labels
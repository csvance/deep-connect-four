import argparse
import os

import numpy as np

from c4_game import C4Game, C4Move, C4ActionResult, C4Team
from c4_model import C4Model


def human_vs_human():
    c4 = C4Game()

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
    c4 = C4Game()
    c4ai = C4Model(epsilon=0., epsilon_min=0., epsilon_decay=1.)
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
            state = c4.perspective_state(current_team)
            move = c4ai.predict(state, valid_moves=valid_moves)
            result = c4.action(move, current_team)

        print(c4.display())

        print("Result: %s" % result)
        if result == C4ActionResult.VICTORY:
            c4.reset()
        elif result == C4ActionResult.TIE:
            c4.reset()


def ai_vs_ai(weights_file: str, epsilon: float, epsilon_decay: float, epsilon_min: float, games: int, gamma: float):
    game_no = 0
    red_wins = 0
    black_wins = 0

    c4 = C4Game()
    c4ai = C4Model(epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma)
    try:
        if weights_file is not None:
            c4ai.load(weights_file)
    except OSError:
        print("Warning: could not load weights file!")
        pass

    while True:
        current_team = c4.current_turn()

        valid_moves = c4.valid_moves()
        if np.sum(valid_moves) == 0:
            print("Tie!")

            # Reset
            c4.reset()
            continue

        state = c4.perspective_state(current_team)
        move = c4ai.predict(state, valid_moves=valid_moves)

        result = c4.action(move, current_team)
        if result == C4ActionResult.VICTORY:
            # Update stats
            if current_team == C4Team.RED:
                red_wins += 1
            elif current_team == C4Team.BLACK:
                black_wins += 1

            # Train
            winning_data = c4.training_data()
            if winning_data is not None:
                loss_count = 0
                loss_sum = 0.
                for state_i, action, reward, state_f, done in winning_data:
                    history = c4ai.train(state_i, action, reward, state_f, done)
                    loss_count += 1
                    loss_sum += history.history['loss'][0]
            else:
                loss_sum = 0.
                loss_count = 1.

            print("Red: %d Black %d Epsilon: %f Loss: %f" % (red_wins, black_wins, c4ai.epsilon, loss_sum / loss_count))
            print(c4.display())
            print("")

            if game_no != 0 and game_no % games == 0:
                print("Saving...")
                c4ai.save(weights_file)
                print("Done.")

            game_no += 1

            # Reset
            c4.reset()


if __name__ == '__main__':
    np.seterr(all='raise')

    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('--weights-file', type=str, default="weights.h5")
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--epsilon-decay', type=float, default=0.9999)
    parser.add_argument('--epsilon-min', type=float, default=0.05)
    parser.add_argument('--training-games', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.mode == 'hvh':
        human_vs_human()
    elif args.mode == 'ava':
        ai_vs_ai(weights_file=args.weights_file, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay,
                 epsilon_min=args.epsilon_min, games=args.training_games, gamma=args.gamma)
    elif args.mode == 'hva':
        human_vs_ai(args.weights_file)
    else:
        print("Valid modes are: ")
        print("hvh - Human vs Human")
        print("hva - Huamn vs AI")
        print("ava - AI vs AI (Training)")

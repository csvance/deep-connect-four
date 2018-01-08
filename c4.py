import argparse
import os

import numpy as np

from c4_game import C4Game, C4MoveResult, C4Team
from c4_model import C4Model


def human_vs_ai(weights_file: str):
    pass


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
        valid_moves = c4.state.valid_moves()

        move = c4ai.predict(c4.state, valid_moves=valid_moves)
        result = c4.action(move)

        if result == C4MoveResult.VICTORY:
            # Update stats
            if current_team == C4Team.RED:
                red_wins += 1
            elif current_team == C4Team.BLACK:
                black_wins += 1

            # Train
            training_data = c4.sample()
            loss_count = 0
            loss_sum = 0.
            for state in training_data:
                history = c4ai.train(state)
                loss_count += 1
                loss_sum += history.history['loss'][0]

            min, max, avg, stdev, med, info_loss = c4ai.stats()

            print("Red: %d Black %d Epsilon: %f Loss: %f" % (
                red_wins, black_wins, c4ai.epsilon, loss_sum / loss_count))
            print("Avg: %f Med: %f Std: %f Destroyed: %f\nRange: [%f, %f]" % (avg, med, stdev, info_loss, min, max))
            print(c4.display())
            print("")

            if game_no != 0 and game_no % games == 0:
                print("Saving...")
                c4ai.save(weights_file)
                print("Done.")

            game_no += 1

            # Reset
            c4.reset()

        elif result == C4MoveResult.TIE:
            print("Tie.")

            # Reset
            c4.reset()


if __name__ == '__main__':
    np.seterr(all='raise')

    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('--weights-file', type=str, default="weights.h5")
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--epsilon-decay', type=float, default=0.99999)
    parser.add_argument('--epsilon-min', type=float, default=0.05)
    parser.add_argument('--training-games', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.75)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.mode == 'ava':
        ai_vs_ai(weights_file=args.weights_file, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay,
                 epsilon_min=args.epsilon_min, games=args.training_games, gamma=args.gamma)
    elif args.mode == 'hva':
        human_vs_ai(args.weights_file)
    else:
        print("Valid modes are: ")
        print("hvh - Human vs Human")
        print("hva - Huamn vs AI")
        print("ava - AI vs AI (Training)")

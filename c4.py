import argparse
import os

import numpy as np
import csv

from c4_game import C4Game, C4MoveResult, C4Team, C4Action
from c4_model import C4Model


def human_vs_ai(weights_file: str):
    c4 = C4Game()
    c4ai = C4Model(epsilon=0., epsilon_min=0.)
    c4ai.load(weights_file)

    print(c4.display())

    while True:
        current_team = c4.current_turn()
        valid_moves = c4.state.valid_moves()

        if current_team == C4Team.BLACK:
            move = input("Move(%s): " % current_team)
            if move == 'q':
                return
            move = C4Action(int(move) - 1)
        elif current_team == C4Team.RED:
            move = c4ai.predict(c4.state, valid_moves=valid_moves)

        result = c4.action(move)

        print(c4.display())
        print("Result: %s" % result)

        if result == C4MoveResult.VICTORY:
            c4.reset()
            print(c4.display())
            continue
        elif result == C4MoveResult.TIE:
            c4.reset()
            print(c4.display())
            continue
        elif result == C4MoveResult.INVALID:
            print(c4.display())
            continue


def ai_vs_ai(weights_file: str, epsilon: float, epsilon_steps: int, epsilon_min: float, games: int,
             gamma: float, gamma_steps: int, gamma_max: float,
             k: int):
    game_no = 0
    red_wins = 0
    black_wins = 0

    headers = ['red_wins', 'black_wins', 'epsilon', 'game_length', 'loss', 'avg', 'med', 'std', 'clipped', 'gamma']
    log_file = open('log.csv', 'w')
    log_writer = csv.DictWriter(log_file, fieldnames=headers)
    log_writer.writeheader()

    c4 = C4Game()
    c4ai = C4Model(epsilon=epsilon, epsilon_steps=epsilon_steps, epsilon_min=epsilon_min,
                   gamma=gamma, gamma_steps=gamma_steps, gamma_max=gamma_max, k=k)
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
            if not c4.duplicate:
                training_data = c4.sample(batch_size=c4.turn + 1)
            else:
                print("Duplicate.")
                training_data = None

            if training_data is not None:

                loss_count = 0
                loss_sum = 0.
                for state in training_data:
                    history = c4ai.train(state)
                    loss_count += 1
                    loss_sum += history.history['loss'][0]

                min, max, avg, stdev, med, clipped, steps = c4ai.stats()

                stats = {}
                stats['red_wins'] = red_wins
                stats['black_wins'] = black_wins
                stats['epsilon'] = c4ai.epsilon.value
                stats['game_length'] = (c4.turn + 1) / 42.
                stats['loss'] = loss_sum / loss_count
                stats['avg'] = avg
                stats['med'] = med
                stats['std'] = stdev
                stats['gamma'] = c4ai.gamma.value
                stats['clipped'] = clipped
                log_writer.writerow(stats)

                print("Red: %d Black %d Steps: %d" % (red_wins, black_wins, steps))
                print("Epsilon: %f Gamma: %f Loss: %f, LR: %f" % (
                    c4ai.epsilon.value, c4ai.gamma.value, loss_sum / loss_count, c4ai.optimizer.lr))
                print("Avg: %f Med: %f Std: %f Clipped: %f\nRange: [%f, %f]" % (avg, med, stdev, clipped, min, max))
                print(c4.display())
                print("")

                if game_no != 0 and game_no % games == 0:
                    print("Saving...")
                    log_file.flush()
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
    parser.add_argument('--epsilon', type=float, default=1.)
    parser.add_argument('--epsilon-steps', type=int, default=100000)
    parser.add_argument('--epsilon-min', type=float, default=0.05)
    parser.add_argument('--training-games', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--gamma-steps', type=int, default=1000000)
    parser.add_argument('--gamma-max', type=float, default=0.99)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.mode == 'ava':
        ai_vs_ai(weights_file=args.weights_file, epsilon=args.epsilon, epsilon_steps=args.epsilon_steps,
                 epsilon_min=args.epsilon_min, games=args.training_games, gamma=args.gamma,
                 gamma_steps=args.gamma_steps,
                 gamma_max=args.gamma_max, k=args.k)
    elif args.mode == 'hva':
        human_vs_ai(args.weights_file)
    else:
        print("Valid modes are: ")
        print("hvh - Human vs Human")
        print("hva - Huamn vs AI")
        print("ava - AI vs AI (Training)")

import argparse
import csv
import os

import numpy as np

from c4_game import C4Game, C4MoveResult, C4Team, C4Action
from c4_model import C4Model


def human_vs_human(weights_file: str):
    c4 = C4Game()
    c4ai = C4Model(epsilon_start=0., epsilon_end=0.)
    c4ai.load(weights_file)

    print(c4.display())

    while True:
        current_team = c4.current_turn()
        valid_moves = c4.state.valid_moves()

        c4ai.print_suggest(c4.state)

        if current_team == C4Team.BLACK:
            move = input("Move(%s): " % current_team)
            if move == 'q':
                return
            move = C4Action(int(move) - 1)
        elif current_team == C4Team.RED:
            move = input("Move(%s): " % current_team)
            if move == 'q':
                return
            move = C4Action(int(move) - 1)

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


def human_vs_ai(weights_file: str):
    c4 = C4Game()
    c4ai = C4Model(epsilon_start=0., epsilon_end=0.)
    c4ai.load(weights_file)

    print(c4.display())

    while True:
        current_team = c4.current_turn()
        valid_moves = c4.state.valid_moves()

        c4ai.print_suggest(c4.state)

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


def ai_vs_best(c4: C4Game, c4ai: C4Model, games: int = 100):
    best_wins = 0
    ai_wins = 0

    ai_scores = []
    best_scores = []

    c4.reset()
    while True:

        if best_wins + ai_wins >= games:
            return best_wins, ai_wins, np.sum(best_scores) / games, np.sum(ai_scores) / games

        current_team = c4.current_turn()
        valid_moves = c4.state.valid_moves()

        if current_team == C4Team.BLACK:
            move = c4.best_action(valid_moves=valid_moves)
        elif current_team == C4Team.RED:
            move = c4ai.predict(c4.state, valid_moves=valid_moves, epsilon=0.05)

        result = c4.action(move)

        if result == C4MoveResult.VICTORY:
            if c4.current_turn() == C4Team.RED:
                ai_wins += 1
            elif c4.current_turn() == C4Team.BLACK:
                best_wins += 1
            ai_scores.append(c4.red_score)
            best_scores.append(c4.black_score)
            print(c4.display())
            c4.reset()
            continue
        elif result == C4MoveResult.TIE:
            print(c4.display())
            c4.reset()
            continue
        elif result == C4MoveResult.INVALID:
            continue


def ai_vs_ai(weights_file: str, epsilon_start: float, epsilon_steps: int, epsilon_end: float, training_steps: int,
             gamma: float):
    game_no = 0
    red_wins = 0
    black_wins = 0

    stats_headers = ['red_wins', 'black_wins', 'epsilon', 'game_length', 'loss', 'mae', 'avg', 'med', 'std']
    stats_log_file = open('stats_log.csv', 'w')
    stats_log_writer = csv.DictWriter(stats_log_file, fieldnames=stats_headers)
    stats_log_writer.writeheader()

    perf_headers = ['ai_win_rate']
    perf_log_file = open('perf_log.csv', 'w')
    perf_log_writer = csv.DictWriter(perf_log_file, fieldnames=perf_headers)
    perf_log_writer.writeheader()

    c4 = C4Game()
    c4ai = C4Model(epsilon_start=epsilon_start, epsilon_steps=epsilon_steps, epsilon_end=epsilon_end,
                   gamma=gamma)
    try:
        if weights_file is not None:
            c4ai.load(weights_file)
            c4.load()
    except OSError:
        print("Warning: could not load weights file!")
        pass

    while True:
        current_team = c4.current_turn()
        valid_moves = c4.state.valid_moves()

        if current_team == C4Team.BLACK:
            move = c4ai.predict(c4.state, valid_moves=valid_moves)
        elif current_team == C4Team.RED:
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
                training_data = c4.sample()
            else:
                print("Duplicate.")
                training_data = None

            if training_data is not None:

                step_count = 0
                loss_sum = 0.
                mae_sum = 0.
                for state in training_data:
                    history = c4ai.train(state)
                    loss_sum += history.history['loss'][0]
                    mae_sum += history.history['mean_absolute_error'][0]
                    step_count += 1

                min, max, avg, stdev, med, steps = c4ai.stats()

                stats = {}
                stats['red_wins'] = red_wins
                stats['black_wins'] = black_wins
                stats['epsilon'] = c4ai.epsilon.value
                stats['game_length'] = (c4.turn + 1) / 42.
                stats['loss'] = loss_sum / step_count
                stats['mae'] = mae_sum / step_count
                stats['avg'] = avg
                stats['med'] = med
                stats['std'] = stdev
                stats_log_writer.writerow(stats)

                print("Red: %d Black %d Steps: %d" % (red_wins, black_wins, steps))
                print("Epsilon: %f Gamma: %f Loss: %f mae: %f" % (
                    c4ai.epsilon.value, c4ai.gamma, loss_sum / step_count, mae_sum / step_count))
                print("Avg: %f Med: %f Std: %f\nRange: [%f, %f]" % (avg, med, stdev, min, max))
                print(c4.display())
                print("")

                if (game_no != 0 and game_no % 100 == 0) or c4ai.steps >= training_steps:
                    print("Saving...")
                    stats_log_file.flush()
                    c4ai.save(weights_file)

                    best_wins, ai_wins, best_score, ai_score = ai_vs_best(c4, c4ai)
                    ai_win_rate = ai_wins / (best_wins + ai_wins)
                    perf_log_writer.writerow(
                        {'ai_win_rate': ai_win_rate})
                    perf_log_file.flush()
                    print("AI won %f%% of games" % ai_win_rate)

                    if c4ai.steps >= training_steps:
                        print("Ran %d steps." % steps)
                        c4.save()
                        return
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
    parser.add_argument('--epsilon-start', type=float, default=1.)
    parser.add_argument('--epsilon-steps', type=int, default=100000)
    parser.add_argument('--epsilon-end', type=float, default=0.05)
    parser.add_argument('--training-steps', type=int, default=2000000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.mode == 'ava':
        ai_vs_ai(weights_file=args.weights_file, epsilon_start=args.epsilon_start, epsilon_steps=args.epsilon_steps,
                 epsilon_end=args.epsilon_end, training_steps=args.training_steps, gamma=args.gamma)
    elif args.mode == 'hva':
        human_vs_ai(args.weights_file)
    elif args.mode == 'hvh':
        human_vs_human(args.weights_file)
    else:
        print("Valid modes are: ")
        print("hvh - Human vs Human")
        print("hva - Huamn vs AI")
        print("ava - AI vs AI (Training)")

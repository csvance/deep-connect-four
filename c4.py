import argparse
import datetime
import os
import numpy as np

from c4_game import C4Game, C4Move, C4ActionResult, C4Team
from c4_model import C4Model, C4FeatureAnalyzer


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
    c4ai = C4Model()
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
            state = c4.state(current_team)
            move = c4ai.predict(state, valid_moves=valid_moves, argmax=True)
            result = c4.action(move, current_team)

        print(c4.display())

        print("Result: %s" % result)
        if result == C4ActionResult.VICTORY:
            c4.reset()
        elif result == C4ActionResult.TIE:
            c4.reset()


def ai_vs_ai(weights_file: str, epsilon: float, epsilon_decay: float, epsilon_min: float):
    time_string = datetime.datetime.now().strftime('%m-%d-%y-%H-%M-%S')
    game_no = 0
    red_wins = 0
    red_win_streak = 0
    red_loss_streak = 0
    black_wins = 0
    black_win_streak = 0
    black_loss_streak = 0

    c4 = C4Game()
    c4ai = C4Model(epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
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

        state = C4FeatureAnalyzer(c4.state(current_team)).analyze()
        move = c4ai.predict([np.array([state[0]]), np.array([state[1]])], valid_moves=valid_moves, argmax=True)

        result = c4.action(move, current_team)
        if result == C4ActionResult.VICTORY:
            # Update stats
            if current_team == C4Team.RED:
                red_wins += 1
                red_win_streak += 1
                black_win_streak = 0
                red_loss_streak = 0
                black_loss_streak += 1
            elif current_team == C4Team.BLACK:
                black_wins += 1
                black_win_streak += 1
                red_win_streak = 0
                black_loss_streak = 0
                red_loss_streak += 1



            # Train
            data, labels = c4.training_data()

            X = [[], []]
            for state in data:
                results = C4FeatureAnalyzer(state).analyze()
                X[0].append(results[0])
                X[1].append(results[1])
            history = c4ai.train([np.array(X[0]), np.array(X[1])], labels)

            print("Red: %d Black %d Epsilon: %f Loss: %f" % (red_wins, black_wins, c4ai.epsilon, history.history['loss'][0]))
            print(c4.display())
            print("")

            # Save Weights every 1000 games
            if game_no % 1000 == 0:
                c4ai.save('weights_' + str(game_no) + '_' + time_string + '.h5')

            game_no += 1

            # Reset
            c4.reset()

            # Decay epsilon
            c4ai.decay()


if __name__ == '__main__':
    np.seterr(all='raise')

    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('--weights-file')
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    parser.add_argument('--epsilon-min', type=float, default=0.01)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.mode == 'hvh':
        human_vs_human()
    elif args.mode == 'ava':
        ai_vs_ai(weights_file=args.weights_file, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay,
                 epsilon_min=args.epsilon_min)
    elif args.mode == 'hva':
        human_vs_ai(args.weights_file)
    else:
        print("Valid modes are: ")
        print("hvh - Human vs Human")
        print("hva - Huamn vs AI")
        print("ava - AI vs AI (Training)")

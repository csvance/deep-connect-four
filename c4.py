import argparse
import datetime
import os
import numpy as np
from multiprocessing import Queue, Pool
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
            state = C4FeatureAnalyzer(c4.state(current_team)).analyze()
            move = c4ai.predict([np.array([state[0]]), np.array([state[1]])], valid_moves=valid_moves,
                                argmax=True)
            result = c4.action(move, current_team)

        print(c4.display())

        print("Result: %s" % result)
        if result == C4ActionResult.VICTORY:
            c4.reset()
        elif result == C4ActionResult.TIE:
            c4.reset()


def ai_vs_ai(weights_file: str, epsilon: float, epsilon_decay: float, epsilon_min: float, games: int):
    time_string = datetime.datetime.now().strftime('%m-%d-%y-%H-%M-%S')
    game_no = 0
    red_wins = 0
    red_win_streak = 0
    red_loss_streak = 0
    black_wins = 0
    black_win_streak = 0
    black_loss_streak = 0

    winning_game_data = []
    winning_game_labels = []

    work_queue = Queue()
    done_queue = Queue()

    def worker_main(w_queue, d_queue):
        while True:
            state, label = w_queue.get()
            data = C4FeatureAnalyzer(state).analyze()
            d_queue.put((data, label))

    worker_pool = Pool(8, worker_main, (work_queue, done_queue))

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

        if current_team == C4Team.RED:
            loss_streak = red_loss_streak
        elif current_team == C4Team.BLACK:
            loss_streak = black_loss_streak

        state = C4FeatureAnalyzer(c4.state(current_team)).analyze()
        move = c4ai.predict([np.array([state[0]]), np.array([state[1]])], valid_moves=valid_moves,
                            argmax=True)

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

            data, labels = c4.training_data()
            winning_game_data.extend(data)
            winning_game_labels.extend(labels)

            print("Red: %d Black %d Epsilon: %f" % (red_wins, black_wins, c4ai.epsilon))
            print(c4.display())
            print("")

            if game_no != 0 and game_no % games == 0:
                X = [[], []]
                Y = []
                work = 0
                for state_idx, state in enumerate(winning_game_data):
                    work_queue.put((state, winning_game_labels[state_idx]))
                    work += 1
                for i in range(0, work):
                    data, label = done_queue.get()
                    X[0].append(data[0])
                    X[1].append(data[1])
                    Y.append(label)
                c4ai.train([np.array(X[0]), np.array(X[1])], np.array(Y))
                c4ai.save('weights_' + time_string + '.h5')
                winning_game_labels = []
                winning_game_data = []

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
    parser.add_argument('--training-games', type=int, default=100)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.mode == 'hvh':
        human_vs_human()
    elif args.mode == 'ava':
        ai_vs_ai(weights_file=args.weights_file, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay,
                 epsilon_min=args.epsilon_min, games=args.training_games)
    elif args.mode == 'hva':
        human_vs_ai(args.weights_file)
    else:
        print("Valid modes are: ")
        print("hvh - Human vs Human")
        print("hva - Huamn vs AI")
        print("ava - AI vs AI (Training)")

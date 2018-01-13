import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.layers import Dense, Flatten, Input, Conv2D, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from c4_game import C4Action, C4ActionResult, C4State


class Ramp(object):
    def __init__(self, start: float, end: float, steps: int, delay: int = 0):
        self.value = start
        self.start = start
        self.end = end
        self.steps = steps
        self.delay = delay

        self._steps_processed = 0

    def step(self, steps: int) -> float:
        self._steps_processed += steps

        if self._steps_processed < self.delay:
            return self.value

        ramp_vertical = self.end - self.start
        ramp_horizontal = self.steps

        m = ramp_vertical / ramp_horizontal
        x = (self._steps_processed - self.delay)
        b = self.start
        y = m * x + b

        if self.start < self.end:
            self.value = min(self.end, y)
        elif self.start > self.end:
            self.value = max(self.end, y)

        return self.value


class C4Model(object):
    def __init__(self, use_gpu=True, epsilon: float = 1., epsilon_steps: int = 500000, epsilon_min=0.05,
                 gamma=0.0, gamma_steps: int = 1000000, gamma_max: float = 0.9, learning_rate=0.001,
                 learning_rate_start=0.0025, k: int = 2):

        self.epsilon = Ramp(start=epsilon, end=epsilon_min, steps=epsilon_steps)
        self.gamma = Ramp(start=gamma, end=gamma_max, steps=gamma_steps)

        self.reward_memory = []
        self.clipped = 0.
        self.k = k
        self.k_self = int(k / 2)
        self.k_enemy = int(k / 2) + (k % 2)
        self.learning_rate = learning_rate
        self.learning_rate_start = learning_rate_start
        self.steps = 0

        input_heights = Input(shape=(7,))
        input_scores = Input(shape=(7, 4, 2))

        # x_1 = input_heights
        #
        # x_2 = input_scores
        # x_2 = Dense(32, activation='relu')(x_2)
        # x_2 = Dense(64, activation='relu')(x_2)
        # x_2 = Dense(64, activation='relu')(x_2)
        # x_2 = Flatten()(x_2)
        #
        # x = concatenate([x_1, x_2])
        # x = Dense(128, activation='relu')(x)

        x = Flatten()(input_scores)

        output = Dense(len(C4Action), activation='linear')(x)

        model = Model(inputs=[input_heights, input_scores], outputs=output)
        self.optimizer = Adam(lr=learning_rate_start)
        model.compile(optimizer=self.optimizer, loss='mse')
        model.summary()
        self._model = model

        if use_gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            set_session(tf.Session(config=config))

    # Data is the game state, labels are the action taken
    def train(self, result: C4ActionResult):

        positive_reward_sum = 0.
        negative_reward_sum = 0.

        if not result.done:

            new_state = result.new_state.copy()
            for k in range(0, self.k):

                # Invert state perspective to match the current agent
                new_state.invert_perspective()

                prediction = self._model.predict(new_state.state_representation())[0]
                reward = np.amax(prediction)
                action = C4Action(np.argmax(prediction))

                # Enemy reward - new_state is the enemies state to act when k is even
                if k % 2 == 0:
                    negative_reward_sum += reward
                # Self Reward - new state is our state to act on when k is odd
                else:
                    positive_reward_sum += reward

                # Apply the action, advance the state
                move_result = new_state.move(action)

            target = result.reward + self.gamma.value * \
                     (positive_reward_sum / self.k_self - 0. * negative_reward_sum / self.k_enemy)

        else:
            target = result.reward

        # Clip reward
        if target > 1.:
            self.clipped += abs(1 - target)
            target = 1.
        elif target < -1.:
            self.clipped += abs(-1 - target)
            target = -1.

        self.reward_memory.append(target)

        target_f = self._model.predict(result.old_state.state_representation())
        target_f[0][result.action.value] = target

        history = self._model.fit(result.old_state.state_representation(), target_f, epochs=1, verbose=0)

        self.epsilon.step(1)
        self.gamma.step(1)

        # Calculate learning rate based on gamma
        if self.gamma.start != self.gamma.end:
            new_learning_rate = (1. - abs(self.gamma.end - self.gamma.value) / abs(
                self.gamma.start - self.gamma.end)) * self.learning_rate + \
                                (abs(self.gamma.end - self.gamma.value) / abs(
                                    self.gamma.start - self.gamma.end)) * self.learning_rate_start
        else:
            new_learning_rate = self.learning_rate

        self.optimizer.lr = new_learning_rate
        self.steps += 1

        return history

    def print_suggest(self, state: C4State):
        predictions = self._model.predict(state.state_representation())[0]

        p_list = []
        for i in range(0, len(predictions)):
            p_list.append([i + 1, predictions[i]])

        p_list.sort(key=lambda x: x[1], reverse=True)

        for p in p_list:
            print("%d: %f" % (p[0], p[1]))

    def predict(self, state: C4State, valid_moves: np.ndarray, epsilon: float = None) -> C4Action:

        if epsilon is None:
            e = self.epsilon.value
        else:
            e = epsilon

        if np.random.rand() <= e:
            potential_moves = []
            for idx in range(0, len(valid_moves)):
                if valid_moves[idx] == 1:
                    potential_moves.append(C4Action(idx))
            return np.random.choice(potential_moves)

        predictions = self._model.predict(state.state_representation())[0]

        # We only want valid moves
        np.place(valid_moves, valid_moves == 0., [-999999.])
        np.place(valid_moves, valid_moves == 1., 0.)

        predictions = predictions + valid_moves

        return C4Action(np.argmax(predictions))

    def stats(self):
        rewards = np.array(self.reward_memory)
        self.reward_memory = []

        clipped = self.clipped
        self.clipped = 0.

        min = np.min(rewards)
        max = np.max(rewards)
        avg = np.average(rewards)
        stdev = np.std(rewards)
        med = np.median(rewards)
        return min, max, avg, stdev, med, clipped, self.steps

    def save(self, path='weights.h5'):
        self._model.save_weights(filepath=path)

    def load(self, path='weights.h5'):
        self._model.load_weights(filepath=path)

import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.layers import Dense, Flatten, Input, concatenate, SeparableConv2D
from keras.models import Model
from keras.optimizers import Adam

from c4_game import C4Action, C4Experience, C4State, C4MoveResult


class Ramp(object):
    def __init__(self, start: float, end: float, steps: int, delay: int = 0):
        self.value = start
        self.start = start
        self.end = end
        self.steps = steps
        self.delay = delay

        if steps == 0:
            self.value = end

        self._steps_processed = 0

    def step(self, steps: int) -> float:
        self._steps_processed += steps

        if self._steps_processed < self.delay:
            return self.value

        ramp_vertical = self.end - self.start
        ramp_horizontal = self.steps

        try:
            m = ramp_vertical / ramp_horizontal
        except ZeroDivisionError:
            self.value = self.end
            return self.end

        x = (self._steps_processed - self.delay)
        b = self.start
        y = m * x + b

        if self.start < self.end:
            self.value = min(self.end, y)
        elif self.start > self.end:
            self.value = max(self.end, y)

        return self.value


class C4Model(object):
    def __init__(self, use_gpu=True, epsilon_start: float = 1., epsilon_steps: int = 1000000, epsilon_end=0.05,
                 gamma=0.9, learning_rate=0.0001):

        self.epsilon = Ramp(start=epsilon_start, end=epsilon_end, steps=epsilon_steps)
        self.gamma = gamma

        self.reward_memory = []

        self.steps = 0

        input_board = Input(shape=(6, 7, 2))
        input_heights = Input(shape=(7,))
        input_scores = Input(shape=(7, 8, 2))

        x_1 = input_board
        x_1 = SeparableConv2D(64, 4, activation='relu')(x_1)
        x_1 = SeparableConv2D(64, 1, activation='relu')(x_1)
        x_1 = Flatten()(x_1)

        x_2 = input_scores
        x_2 = Dense(64, activation='relu')(x_2)
        x_2 = Dense(64, activation='relu')(x_2)
        x_2 = Flatten()(x_2)

        x_3 = input_heights

        x = concatenate([x_1, x_2, x_3])
        x = Dense(256, activation='relu')(x)

        output = Dense(len(C4Action), activation='linear')(x)

        model = Model(inputs=[input_board, input_scores, input_heights], outputs=output)
        model.compile(optimizer=Adam(lr=learning_rate), loss='mse', metrics=['mae'])
        model.summary()
        self._model = model

        if use_gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            set_session(tf.Session(config=config))

    # Data is the game state, labels are the action taken
    def train(self, experience: C4Experience):

        if not experience.terminal:
            new_state = experience.new_state.copy()

            # Invert state perspective to match the enemy agent
            new_state.invert_perspective()

            # We don't care about the enemy reward, just the most likely action
            prediction = self._model.predict(new_state.state_representation())[0]
            action = C4Action(np.argmax(prediction))

            # Apply the enemy action to advance the state, invert the perspective to match friendly agent
            move_result = new_state.move(action)
            if move_result != C4MoveResult.VICTORY:
                new_state.invert_perspective()

                # Here is where our discounted future reward comes from (next friendly state in our perspective)
                prediction = self._model.predict(new_state.state_representation())[0]

                # Calculate discounted future reward
                target = experience.reward + self.gamma * np.amax(prediction)
            elif C4MoveResult.VICTORY:
                target = -1.
            else:
                target = 0.
        else:
            target = experience.reward

        # Clip reward
        if target > 1.:
            target = 1.
        elif target < -1.:
            target = -1.

        self.reward_memory.append(target)

        target_f = self._model.predict(experience.old_state.state_representation())
        target_f[0][experience.action.value] = target

        history = self._model.fit(experience.old_state.state_representation(), target_f, epochs=1, verbose=0)

        self.epsilon.step(1)

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

        min = np.min(rewards)
        max = np.max(rewards)
        avg = np.average(rewards)
        stdev = np.std(rewards)
        med = np.median(rewards)
        return min, max, avg, stdev, med, self.steps

    def save(self, path='weights.h5'):
        self._model.save_weights(filepath=path)

    def load(self, path='weights.h5'):
        self._model.load_weights(filepath=path)

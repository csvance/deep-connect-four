from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Input, Reshape, concatenate
from keras.models import Model
from keras.optimizers import Adam

from c4_game import C4Move


class C4Model(object):
    def __init__(self, use_gpu=True, epsilon: float = 0., epsilon_decay: float = 0.9999, epsilon_min=0.05,
                 gamma=0.95, learning_rate=0.001):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        input = Input(shape=(6, 7, 1))

        x_1 = Conv2D(1, (1, 4), activation='relu')(input)
        x_2 = Conv2D(1, (4, 1), activation='relu')(input)

        x_1 = Flatten()(x_1)
        x_2 = Flatten()(x_2)

        x_1 = Dense(6, activation='relu')(x_1)
        x_2 = Dense(7, activation='relu')(x_2)

        x_1 = Dense(6, activation='relu')(x_1)
        x_2 = Dense(7, activation='relu')(x_2)

        c = concatenate([x_1, x_2])

        x = Dense(6 * 7, activation='relu')(c)
        output = Dense(len(C4Move), activation='linear')(x)

        model = Model(inputs=input, outputs=output)
        model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
        model.summary()
        self._model = model

        if use_gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            set_session(tf.Session(config=config))

    # Data is the game state, labels are the action taken
    def train(self, state_i: np.ndarray, action: C4Move, reward: float, state_f: np.ndarray, done: bool):

        if not done:
            target = reward + self.gamma * np.amax(self._model.predict(np.array([state_f]))[0])
        else:
            target = reward

        target_f = self._model.predict(np.array([state_i]))
        target_f[0][action] = target

        history = self._model.fit(np.array([state_i]), target_f, epochs=1, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return history

    def predict(self, state, valid_moves: np.ndarray) -> C4Move:
        if np.random.rand() <= self.epsilon:
            potential_moves = []
            for idx in range(0, len(valid_moves)):
                if valid_moves[idx] == 1:
                    potential_moves.append(C4Move(idx))
            return np.random.choice(potential_moves)

        predictions = self._model.predict(np.array([state]))[0]

        # We only want valid moves
        predictions = predictions * valid_moves

        return C4Move(np.argmax(predictions))

    def save(self, path='weights.h5'):
        self._model.save_weights(filepath=path)

    def load(self, path='weights.h5'):
        self._model.load_weights(filepath=path)

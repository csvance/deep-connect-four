import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.layers import Dense, Flatten, Conv2D, Input
from keras.models import Model
from keras.optimizers import Adam

from c4_game import C4Action, C4ActionResult, C4State


class C4Model(object):
    def __init__(self, use_gpu=True, epsilon: float = 0., epsilon_decay: float = 0.99999, epsilon_min=0.05,
                 gamma=0.5, learning_rate=0.001, k: int = 10):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.reward_memory = []
        self.clipped = 0.
        if k % 2 != 0:
            raise ValueError('k must be an even number')
        self.k = k

        input = Input(shape=(6, 7, 2))

        # x = Conv2D(64, (4, 4), strides=1, activation='relu')(input)
        x = Dense(256, activation='relu')(input)
        x = Dense(256, activation='relu')(x)
        x = Flatten()(x)
        # x = Dense(64 * 3 * 4, activation='relu')(x)

        output = Dense(len(C4Action), activation='linear')(x)

        model = Model(inputs=input, outputs=output)
        model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
        model.summary()
        self._model = model

        if use_gpu:
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = True
            set_session(tf.Session(config=config))

    # Data is the game state, labels are the action taken
    def train(self, result: C4ActionResult):

        positive_reward_sum = 0.
        negative_reward_sum = 0.

        if not result.done:

            new_state = result.new_state.copy()
            for k in range(0, self.k):

                prediction = self._model.predict(np.array([new_state.one_hot()]))[0]
                reward = np.amax(prediction)
                action = C4Action(np.argmax(prediction))

                # Enemy reward - new_state is the enemies state to act when k is even
                if k % 2 == 0:
                    negative_reward_sum += reward
                # Self Reward - new state is our state to act on when k is odd
                else:
                    positive_reward_sum += reward

                # Apply the action
                move_result = new_state.move(action)

                # Advance state one turn to change the perspective
                new_state.next_turn()

            target = result.reward + self.gamma * \
                     ((positive_reward_sum / (self.k / 2.)) - (negative_reward_sum / (self.k / 2.)))
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

        target_f = self._model.predict(np.array([result.old_state.one_hot()]))
        target_f[0][result.action.value] = target

        history = self._model.fit(np.array([result.old_state.one_hot()]), target_f, epochs=1, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return history

    def predict(self, state: C4State, valid_moves: np.ndarray) -> C4Action:
        if np.random.rand() <= self.epsilon:
            potential_moves = []
            for idx in range(0, len(valid_moves)):
                if valid_moves[idx] == 1:
                    potential_moves.append(C4Action(idx))
            return np.random.choice(potential_moves)

        predictions = self._model.predict(np.array([state.one_hot()]))[0]

        # We only want valid moves
        predictions = predictions * valid_moves

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
        return min, max, avg, stdev, med, clipped

    def save(self, path='weights.h5'):
        self._model.save_weights(filepath=path)

    def load(self, path='weights.h5'):
        self._model.load_weights(filepath=path)

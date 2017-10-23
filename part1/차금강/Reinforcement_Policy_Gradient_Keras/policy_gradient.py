import copy
import numpy as np
from game import Game
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODE = 20000000

class REINFORCEMENTAgent:
    def __init__(self):
        self.load_model = True
        self.action_space = [0,1,2]
        self.action_size = len(self.action_space)
        self.state_size = 60
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights('reinforce.h5')

    def build_model(self):
        model = Sequential()
        model.add(Dense(60, input_dim=self.state_size, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()

        return model

    def build_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        discounted_rewards = K.placeholder(shape=[None, ])
        action_prob = K.sum(action*self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, action, discounted_rewards], [], updates=updates)

        return train

    def get_action(self, state):
        policy = self.model.predict(state)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []


if __name__=="__main__":
    game = Game(6,10,show_game="False")
    agent = REINFORCEMENTAgent()
    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODE):
        done = False
        score = 0
        state = game.reset()
        state = np.reshape(state, [1, 60])

        while not done:
            global_step += 1

            action = agent.get_action(state)
            next_state, reward, done = game.step(action)
            next_state = np.reshape(next_state, [1, 60])
            agent.append_sample(state, action, reward)

            score += reward
            state = copy.deepcopy(next_state)
            if done:
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "score:", score, "time_step:", global_step)
                global_step = 0
            if e%10 == 0:
                agent.model.save_weights("reinforce.h5")
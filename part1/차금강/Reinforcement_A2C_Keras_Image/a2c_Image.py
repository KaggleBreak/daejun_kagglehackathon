import sys
from game import Game
import pylab
import numpy as np
from keras.layers import Dense, Reshape, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K

EPISODES = 100000


class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = True
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()

        if self.load_model:
            self.actor.load_weights("cartpole_actor_trained.h5")
            self.critic.load_weights("cartpole_critic_trained.h5")

    def build_actor(self):
        actor = Sequential()
        actor.add(Reshape((6,10,1), input_shape=(self.state_size, )))
        actor.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        actor.add(Flatten())
        actor.add(Dense(24, input_dim=self.state_size, activation='relu'))
        actor.add(Dense(24, activation='relu'))
        actor.add(Dense(24, activation='relu'))
        actor.add(Dense(24, activation='relu'))
        actor.add(Dense(24, activation='relu'))
        actor.add(Dense(self.action_size, activation='softmax'))
        return actor

    def build_critic(self):
        critic = Sequential()
        critic.add(Reshape((6,10,1), input_shape=(self.state_size, )))
        critic.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        critic.add(Flatten())
        critic.add(Dense(24, input_dim=self.state_size, activation='relu'))
        critic.add(Dense(24, activation='relu'))
        critic.add(Dense(24, activation='relu'))
        critic.add(Dense(24, activation='relu'))
        critic.add(Dense(self.value_size, activation='linear'))
        critic.summary()
        return critic

    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [],
                           updates=updates)
        return train

    def critic_optimizer(self):
        target = K.placeholder(shape=[None, ])

        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)

        return train

    def train_model(self, state, action, reward, next_state, done):
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value

        self.actor_updater([state, act, advantage])
        self.critic_updater([state, target])


if __name__ == "__main__":
    env = Game(6,10,show_game="False")
    state_size = 60
    action_size = 3

    agent = A2CAgent(state_size, action_size)
    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            global_step += 1
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                agent.actor.save_weights("cartpole_actor_trained.h5")
                agent.critic.save_weights("cartpole_critic_trained.h5")
                scores.append(score)
                episodes.append(e)
                #pylab.plot(episodes, scores, 'b')
                #pylab.savefig("./save_graph/cartpole_a2c.png")
                print("episode:", e, "  score:", global_step)
                global_step = 0
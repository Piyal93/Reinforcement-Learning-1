
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 10
MAX_SKIPS = 4
EPISODES = 2500
MOVE_PENALTY = 1
END_REWARD = 0
epsilon = 0.9
EPS_DECAY = 0.9998  # the rate at which epsilon decays
SHOW_EVERY = 30  # how often to the environment is shown i.e. after every 30 episodes.

start_q_table = None # Innitial Q- Table

LEARNING_RATE = 0.1
DISCOUNT = 0.95

class Utils:
    def __init__(self):
        self.x = 0

    def __str__(self):
        return f"{self.x}"

    def __sub__(self, other):
        return (self.x-other.x)

    def action(self, choice):
        '''
        Gives us 4 total skip options. (1,2,3 and 4)
        '''
        if choice == 0:
            self.move(x=1)
        elif choice == 1:
            self.move(x=2)
        elif choice == 2:
            self.move(x=3)
        elif choice == 3:
            self.move(x=4)

    def move(self, x=False):
        """
        Gives the position after the action taken (the next state) 
        """

        # If no value for x, don't move
        if not x:
            self.x += 0
        else:
            self.x += x
        #If the position of the person is out of bounds
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = 0

if start_q_table is None:
    # initialize the q-table with random values
    q_table = {}
    for i in range(-SIZE+1, 1):
        q_table[(i)] = [np.random.uniform(-5, 0) for i in range(4)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []
epsilon_list = []
for episode in range(EPISODES):
    player = Utils()
    top = SIZE-1
    k = np.random.randint(1, MAX_SKIPS) # Action space or no. of skips max 4 and min 1

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} episode mean reward: {np.mean(episode_rewards[-SHOW_EVERY:])}")

    episode_reward = 0
    for i in range(200):
        obs = (player.x-top)
        if np.random.random() > epsilon:
            # GET THE EXPLOITATION ACTION
            action = np.argmax(q_table[obs])
        else:
            # GET THE EXPLORATION ACTION
            action = np.random.randint(0, k+1)
        # Taking the action
        player.action(action)

        if player.x == top :
            reward = END_REWARD
        else:
            reward = -MOVE_PENALTY
        new_obs = (player.x-top)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == END_REWARD:
            new_q = END_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q
        episode_reward += reward
        if reward == END_REWARD:
            break

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon_list.append(epsilon)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Moving Average of Reward for every {SHOW_EVERY} episodes")
plt.xlabel("episode Number")
plt.show()
plt.savefig('plot1.png')

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)

moving_avg_ep = np.convolve(epsilon_list, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot(moving_avg, moving_avg_ep)
plt.ylabel(f"Epsilon Value after every {SHOW_EVERY} episodes")

plt.xlabel(f"Moving Average of Reward for every {SHOW_EVERY} episodes")
plt.show()
plt.savefig('plot1.png')



import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 10

EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998  # the rate at which epsilon decays
SHOW_EVERY = 3000  # how often to the environment is shown i.e. after every 30 episodes.

start_q_table = None # Innitial Q- Table

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        """
        Gives the position of the player after the action taken (the next state) 
        """

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y


        # If the player hits the wall or out of bounds
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


if start_q_table is None:
    # initialize the q-table with random values
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                    for y2 in range(-SIZE+1, SIZE):
                        q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []
epsilon_list = []

for episode in range(EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} episode mean reward: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE EXPLOITATION ACTION
            action = np.argmax(q_table[obs])
        else:
            # GET THE EXPLORATION ACTION
            action = np.random.randint(0, 4)
        # Taking the action
        player.action(action)


        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            """
            Designing the environment having 10x10 grid, one blue player(movable), 
            1 red enemy(non-movable) and 1 green food(non-movable)
            """
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # Grid of 10x10x3
            env[food.x][food.y] = d[FOOD_N]  # Green Food
            env[player.x][player.y] = d[PLAYER_N]  # Blue Player
            env[enemy.x][enemy.y] = d[ENEMY_N]  # red Enemy
            img = Image.fromarray(env, 'RGB')  
            img = img.resize((300, 300))  
            cv2.imshow("image", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # End the game/episode if player eats the food or get hit by the enemy
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
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


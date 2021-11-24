
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import warnings
import requests 
import time
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from tensorflow.python.util import memory
warnings.filterwarnings("ignore")
from rl.memory import SequentialMemory



a = [[[1, 2, 3]], [[1], [2, 3]], [[3], [1, 2]], [[1], [2], [3]]]

action_namelist=[]
action_value_list=[]

def create_the_action_plan(a):
    for i in range(len(a)):
        action_namelist.append("{}".format(i))
    for element in a:
        action_value_list.append(element)
    zip_iterator = zip(action_namelist, action_value_list) 
    action_dictionary = dict(zip_iterator)
    return action_namelist, action_value_list, action_dictionary


def initial_state(actions_):
    len_actions=[]
    for action in actions_:
        len_actions.append(len(action))
    idx_initial_action = max(len_actions)
    return actions_[idx_initial_action]

def trigger_utility():
    #triggers the utility function to calculate the value 
    url= 'http://127.0.0.1:5003/utility'
    response = requests.get(url, timeout=2.50)

    return(float(response.text))

def create_index_(action_value_list:list):
    actions_keys=[]
    counter_=0
    for iteration in range(len(action_value_list)-1):
        actions_keys.append(counter_)
        counter_+=1

    return actions_keys

def cost_calculation(reward, seen_actions):
    if len(seen_actions[-1]) < len(seen_actions[-2]):
        #print("Agent must destroy a vm")
        destroy = 10
        create = 0
        reward = reward+destroy+create 
    elif len(seen_actions[-1]) == len(seen_actions[-2]):
        #print('Agent has nothing to do')
        destroy = 0
        create=0
        reward = reward+destroy+create 
    else:
        #print("Agent must create a vm")
        destroy =0
        create=-10
        reward = reward+destroy+create 

    return reward


def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

class Application_Env(Env):
    
    def __init__(self,action_value_list,utility, action_dictionary, actions_keys:list):
        self.action_space = Discrete(len(action_value_list)-1) # number of possible actions
        self.observation_space = Box(low = np.array([0.1]), high = np.array([0.99])) #utility function value boundaries
        self.all_state = action_value_list
        self.list_with_all_possible_actions = action_value_list[:-1]
        self.initial_property = action_value_list[-1]
        self.utility = utility
        self.list_of_utilities = [float(self.utility)]
        self.seen_actions = [self.initial_property]
        self.action_dict = action_dictionary
        self.actions_keys= actions_keys
        self.seen_action_keys = []
    
    def step(self, action):

        self.seen_action_keys.append(action)
        translated_action  = self.list_with_all_possible_actions[action]
        self.seen_actions.append(translated_action)
        print("the agent chooses the action {}".format(action))
        #print("seen actions : {}".format(self.seen_actions))
        self.utility = trigger_utility()
        self.list_of_utilities.append(self.utility)
        print(self.utility)
        print(self.seen_action_keys)

        if len(self.seen_action_keys) != len(set(self.seen_action_keys)):
            
            counter=0
            reward =0
            self.seen_action_keys = self.actions_keys[:-1]
            self.seen_actions = self.seen_actions[:-1]
            
        else:

            if self.list_of_utilities[-1] >=0.8 and self.list_of_utilities[-1]>self.list_of_utilities[-2]:
                reward_ =100
                reward = cost_calculation(reward_, self.seen_actions)         

            elif self.list_of_utilities[-1]>0.6 and self.list_of_utilities[-1]>self.list_of_utilities[-2]:
                reward_=5
                reward = cost_calculation(reward_, self.seen_actions) 
            
            elif self.list_of_utilities[-1]>0.6 and self.list_of_utilities[-1] < self.list_of_utilities[-2]: 
                reward_=1
                reward = cost_calculation(reward_, self.seen_actions) 

            elif self.list_of_utilities[-1]<0.6:
                reward_=-1
                reward = cost_calculation(reward_, self.seen_actions) 

            
            counter = 1 if (self.list_of_utilities[-1] >=0.8 and self.list_of_utilities[-2]<self.list_of_utilities[-1]) else 0


        if counter != 1 and not len(self.seen_actions) == len(self.all_state):
            #print("is not done")
            done=False

        elif counter !=1 and len(self.seen_actions) == len(self.all_state):
            #print("is done")
            done=True
            self.seen_actions= [self.initial_property]
            self.seen_action_keys=[]

        elif counter ==1:
            #print("is done")
            done=True
            self.seen_actions= [self.initial_property]
            self.seen_action_keys=[]
            #set a placeholoder for info

        info={}
        #print("reward is: {}".format(reward))
        #print("************************")
        return self.utility, reward, done, info

    def reset(self):
        self.utility = trigger_utility()


        return self.utility



int_act = initial_state(a)
num_of_actions = 2
action_namelist, action_value_list, action_dictionary = create_the_action_plan(a)
utility = trigger_utility() #is the utility for every running component in seperate vms
actions_keys = create_index_(action_value_list)
my_env = Application_Env(action_value_list=action_value_list,utility=utility, 
                                    action_dictionary=action_dictionary, actions_keys=actions_keys)

episodes=2
for episode in range(episodes+1):
        initial_utility = my_env.reset()
        done =False
        score=0

        while not done:
            action = my_env.action_space.sample()
            utility, reward, done, info = my_env.step(action)
            score+=reward
            #time.sleep(2)

            #print("------------")
            print("episode: {}, score: {}".format(episode, score))

actions = my_env.action_space.n
states = my_env.observation_space.shape

model = build_model(states, actions)

model.summary()

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(my_env, nb_steps=100, visualize=False, verbose=1)



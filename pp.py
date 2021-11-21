
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
    url= 'http://127.0.0.1:5001/utility'
    response = requests.get(url, timeout=2.50)

    return(float(response.text))

def create_index_(action_value_list:list):
    actions_keys=[]
    counter_=0
    for iteration in range(len(action_value_list)-1):
        actions_keys.append(counter_)
        counter_+=1

    return actions_keys

def cost_calculation(action,actions_keys:list,action_value_list:list,seen_actions:list, seen_actions_keys:list):

    idx_action_ = actions_keys.index(action)
    translated_action = action_value_list[idx_action_]

    #print("The agent chooses the action {} which is translated in: {}".format(action, translated_action))
    if len(translated_action) < len(seen_actions[-1]):
        #print("Agent must destroy a vm")
        destroy = 10
        create = 0
        seen_actions.append(translated_action)
        seen_actions_keys.append(action)
    elif len(translated_action) == len(seen_actions[-1]):
        #print('Agent has nothing to do')
        destroy = 0
        create=0
        seen_actions.append(translated_action)
        seen_actions_keys.append(action)
    else:
        #print("Agent must create a vm")
        destroy =0
        create=-10
        seen_actions.append(translated_action)
        seen_actions_keys.append(action)
    
    return destroy, create, seen_actions, seen_actions_keys


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
        self.seen_action_keys = [0]

    def step(self, action): 
        #print("Agent tests the {} action".format(action))

        self.utility = trigger_utility()
        
        #print("The agent selects to group components as {} and the utility function is: {}".format(action, self.utility))
        
        destroy, create, self.seen_actions, self.seen_actions_keys = cost_calculation(action,self.actions_keys, self.list_with_all_possible_actions, self.seen_actions, self.seen_action_keys)
        
        if self.utility>0.8 and self.utility> self.list_of_utilities[-1]:
            reward=100+destroy+create
            self.list_of_utilities.append(self.utility)
            counter=1

        elif self.utility>0.6 and self.utility> self.list_of_utilities[-1]:
            reward=1+destroy+create
            self.list_of_utilities.append(self.utility)
            counter = 0

        elif self.utility>0.6 and self.utility<= self.list_of_utilities[-1]:
            reward = 0+destroy+create
            self.list_of_utilities.append(self.utility)
            counter = 0

        elif self.utility<0.6:
            reward=-1+destroy+create
            self.list_of_utilities.append(self.utility)
            counter = 0

        if counter != 1:
            done=False
        else:
            done=True
            #set a placeholoder for info
        info={}

        return self.utility, reward, done, info

    def render(self):
        #useful for visualization 
        pass

    def reset(self):
        #set initial state and the utility
        return self.utility

    def reset_seen_actions(self):
        self.seen_actions = [self.initial_property]
        return self.seen_actions

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(units=24, activation="relu", input_shape=states))
    model.add(Dense(units=24, activation="relu"))
    model.add(Dense(units=actions, activation="linear"))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


if __name__ == "__main__":


    int_act = initial_state(a)
    num_of_actions = 2
    action_namelist, action_value_list, action_dictionary = create_the_action_plan(a)
    utility = trigger_utility() #is the utility for every running component in seperate vms
    actions_keys = create_index_(action_value_list)
    my_env = Application_Env(action_value_list=action_value_list,utility=utility, 
                                    action_dictionary=action_dictionary, actions_keys=actions_keys)
    episodes=30
    for episode in range(episodes+1):
            initial_utility = my_env.reset()
            done =False
            score=0
            counter_=0
            c_ = 0
            seen_actions = my_env.reset_seen_actions()
            #print("seen actions {}".format(seen_actions))
            lr = random.sample(my_env.actions_keys, len(my_env.actions_keys))

            while not done and c_ <len(my_env.actions_keys):
                action=lr[c_]
                utility, reward, done, info = my_env.step(action)
                #print(reward)
                #print(utility)
                #time.sleep(2)
                score+=reward
                c_+=1
                time.sleep(2)
                #print("***********************")
            #print("------------")
            print("episode: {}, score: {}".format(episode, score))
            print(seen_actions)

    actions = my_env.action_space.n
    states = my_env.observation_space.shape
    

    model = build_model(states=states, actions=actions)
    model.summary()


    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(my_env, nb_steps=50000, visualize=False, verbose=1)


#my_env=Application_Env(action_dictionary,int_act,first_response)
#print(my_env.all_state)
'''

my_env = Application_Env(action_value_list=action_value_list,int_act=int_act, first_response=first_response)

print(type(my_env.property_list))
'''





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
    for iteration in range(len(action_value_list)):
        actions_keys.append(counter_)
        counter_+=1
    return actions_keys

def cost_calculation(reward, seen_actions):
    if len(seen_actions[-1]) < len(seen_actions[-2]):
        print("Agent must destroy vm/vms")
        destroy = 10
        create = 0
        reward = reward+destroy+create 
    elif len(seen_actions[-1]) == len(seen_actions[-2]):
        print('Agent has nothing to do')
        destroy = 0
        create=0
        reward = reward+destroy+create 
    else:
        print("Agent must create  vm/vms")
        destroy =0
        create=-10
        reward = reward+destroy+create 

    return reward

def create_map(combinations,action_dictionary:dict ):
    list_of_actions=[]
    list_of_utilities=[]
    list_of_index=[]
    list_of_labeled_actions=[]
    for element in combinations:
        for i in range(1,11):
            list_of_actions.append(element)
            list_of_utilities.append(trigger_utility())
            time.sleep(2)

    for i in range(1,41):
        list_of_index.append(i)
    
    for k, v in action_dictionary.items():
        for iteration in list_of_actions:
            if iteration == v:
                list_of_labeled_actions.append(k)
    #print(list_of_combinations)

    for iteration in range(len(list_of_labeled_actions)):
        list_of_labeled_actions[iteration] = int(list_of_labeled_actions[iteration])

    return list_of_actions, list_of_index, list_of_utilities, list_of_labeled_actions

def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def act(action, seen_action_keys:list, list_of_labeled_actions:list, index_actions:list,utility_list:list):
    action_that_must_be_used=[]
    utilities_that_must_be_used=[]
    indeces_that_must_be_used=[]
    action_that_must_not_used=[]
    utilities_that_must_not_used=[]
    index_that_must_not_used=[]
    
    if action  == seen_action_keys[-1]: #if not the same action with the previous one 
        duplicates_list_actions = duplicates(list_of_labeled_actions, seen_action_keys[-1])

        for elements in duplicates_list_actions:
            action_that_must_not_used.append(list_of_labeled_actions[elements])
            utilities_that_must_not_used.append(utility_list[elements])
            index_that_must_not_used.append(list_of_index[elements])

        available_actions = [item for item in list_of_labeled_actions if item not in action_that_must_not_used] 

        available_utilities = [item for item in utility_list if item not in utilities_that_must_not_used]   
        available_indeces_ = [item for item in list_of_index if item not in index_that_must_not_used]   

        index_ = random.choice(available_indeces_)
        idx_ = available_indeces_.index(index_)
        action_  = available_actions[idx_]
        utility = available_utilities[idx_]
    
    else: #if not the same action with the previous one 

        duplicates_list_actions = duplicates(list_of_labeled_actions, action)
        #print(duplicates_list_actions)
        for element in duplicates_list_actions:
            action_that_must_be_used.append(list_of_labeled_actions[element])
            utilities_that_must_be_used.append(utility_list[element])
            indeces_that_must_be_used.append(list_of_index[element])

        available_actions = [item for item in list_of_labeled_actions if item in action_that_must_be_used] 
        available_utilities = [item for item in utility_list if item in utilities_that_must_be_used]   
        available_indeces = [item for item in list_of_index if item in indeces_that_must_be_used]

        index_ = random.choice(available_indeces)
        idx_ = available_indeces.index(index_)
        action_  = available_actions[idx_]
        utility = available_utilities[idx_]

    return action_ , utility, idx_

def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=100, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

class Application_Env(Env):

    def __init__(self,list_of_actions,list_of_utilities,list_of_index,action_value_list, list_of_labeled_actions,action_dictionary, actions_keys):
        self.action_space = Discrete(len(action_value_list)) # number of possible actions
        self.observation_space = Box(low = np.array([0.1]), high = np.array([0.99])) #utility function value boundaries
        self.action_value_list = action_value_list
        self.all_state = list_of_utilities
        self.list_with_all_possible_actions = list_of_actions
        self.initial_property = [list_of_actions[0]]
        self.utility = [list_of_utilities[0]]
        self.list_of_utilities = self.utility
        self.seen_actions = self.initial_property
        self.action_dict = action_dictionary
        self.actions_keys= actions_keys
        self.indices = list_of_index
        self.seen_indices = [list_of_index[0]]
        self.action_labels = list_of_labeled_actions
        self.seen_action_keys = [list_of_labeled_actions[0]]

    def step(self, action):
        action, utility,idx_ = act(action,seen_action_keys=self.seen_action_keys,list_of_labeled_actions=self.action_labels,index_actions=self.indices,
                                 utility_list=self.all_state)
        print(utility)

        self.seen_action_keys.append(action)
        idx_seen_action = self.actions_keys.index(self.seen_action_keys[-1])
        translated_action  = self.action_value_list[idx_seen_action]
        self.seen_actions.append(translated_action)

        #print("the agent chooses the action {} which is translated in action: {} ".format(action,translated_action))
        #print("seen actions : {}".format(self.seen_actions))

        self.list_of_utilities.append(utility)

        if self.list_of_utilities[-1] >=0.9 and self.list_of_utilities[-1]>self.list_of_utilities[-2]:
            reward_ =100
            reward = cost_calculation(reward_, self.seen_actions)  
        
        elif self.list_of_utilities[-1]>=0.9 and self.list_of_utilities[-1]<self.list_of_utilities[-2]:
            reward_ = 70
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

        time.sleep(3)   
        counter = 1 if (self.list_of_utilities[-1] >=0.9) else 0

        if  counter ==1:
                done=True
                self.seen_actions= [self.initial_property]
                self.seen_action_keys=[0]
        else:
            done=False
            pass

        #set a placeholoder for info

        info={}
        #print("reward is: {}".format(reward))
        #print("************************")
        return utility, reward, done, info

    def reset(self):
        self.utility = list_of_utilities[0]
        return self.utility

if __name__ == "__main__":

    action_namelist, action_value_list, action_dictionary = create_the_action_plan(a)
    list_of_actions, list_of_index, list_of_utilities, list_of_labeled_actions = create_map(combinations=a, action_dictionary=action_dictionary)
    actions_keys = create_index_(action_value_list)
    my_env = Application_Env(list_of_actions=list_of_actions,action_value_list=action_value_list,list_of_utilities=list_of_utilities, 
                            list_of_index=list_of_index,list_of_labeled_actions=list_of_labeled_actions, 
                            action_dictionary=action_dictionary, actions_keys=actions_keys)


    episodes=10
    for episode in range(episodes+1):
            utility = my_env.reset()
            done =False
            score=0
            my_env.reset()
            while not done:
                action = my_env.action_space.sample()
                utility, reward, done, info = my_env.step(action)
                score+=reward


            print("------------")
            print("episode: {}, score: {}".format(episode, score))
            print("------------")

            actions = my_env.action_space.n
            states = my_env.observation_space.shape

            model = build_model(states, actions)

            model.summary()

            dqn = build_agent(model, actions)
            #dqn.compile(Adam(lr=1e-3), metrics=['mae'])
            #dqn.fit(my_env, nb_steps=100, visualize=False, verbose=1)

            scores = dqn.test(my_env, nb_episodes=100, visualize=False)
            print(np.mean(scores.history['episode_reward']))


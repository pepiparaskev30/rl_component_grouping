import tests
from pyvis.network import Network
import networkx as nx
from itertools import combinations
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
import requests
import time


action_namelist=[]
action_value_list=[]

def create_the_action_plan(x):
    for i in range(len(x)):
        action_namelist.append("{}".format(i))
    for element in x:
        action_value_list.append(element)
    zip_iterator = zip(action_namelist, action_value_list) 
    action_dictionary = dict(zip_iterator)
    
    return action_namelist, action_value_list, action_dictionary

def rSubset(arr, r):
    """
    Returns: the list of all subsets of length r
    """
    return list(combinations(arr, r))

def initial_state(actions_):
    """
    Returns: the initial agent action:  All the application components in seperate vms
    """
    len_actions=[]
    for action in actions_:
        len_actions.append(len(action))

    idx_initial_action = max(len_actions)

    return actions_[idx_initial_action]


def create_application_graph_components(action_namelist,action_valuelist):
    """
    This function creates and visualizes the graph with all possible grouping combinations

    Args:
        action_namelist: the actions names
        action_valuelist: the actions labels
    Returns:
        an html viz graph 
    """
    action_namelist_ = list(map(int, action_namelist)) # convert string elements in the list to integers
    edge_plan_combinations = rSubset(action_namelist_,2)   #create all the node association combinations 
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white') #create the graph 
    #create the number of nodes and set their labels
    for node_ in range(len(action_value_list)):
        net.add_node(node_, label = "{}".format(action_value_list[node_]))
    #create the edges 
    for i in range(len(edge_plan_combinations)):
        net.add_edge(edge_plan_combinations[i][0],edge_plan_combinations[i][1])
    # and visualize
    return net.show('nx.html')
    
def report_of_app_components(x_combinations):
    """
    This function returns a brief report about the application components
    and its combinations regarding grouping
    """
    print("The given application has {} components \n and the possible combonations are: {}".format(len(x_combinations), x_combinations))


def initial_state(actions_):
    '''
    This function returns the initial state of application 
    components: each component runs in a single vm i.e. [1], [2], [3]
    
    Args: 
        actions_ : all the possible component combinations
    
    Returns: 
        the initial state of application 
    '''
    len_actions=[]
    for action in actions_:
        len_actions.append(len(action))

    idx_initial_action = max(len_actions)

    return actions_[idx_initial_action]


def trigger_utility():
    """
    This function triggers the utility function and returns the 
    utility value for the specific deployment plan 
    """
    #triggers the utility function to calculate the value 
    url= 'http://127.0.0.1:5001/utility' # we have created a simple flask app that returns random numbers as the utility value
    response = requests.get(url, timeout=2.50)

    return(float(response.text))

def cost_calculation(action, seen_actions):
    if len(action) < len(seen_actions[-1]):
        print("Agent must destroy a vm")
        destroy = 10
        create = 0
        seen_actions.append(action)
    elif len(action) == len(seen_actions[-1]):
        print('Agent has nothing to do')
        destroy = 0
        create=0
        seen_actions.append(action)
    else:
        print("Agent must create a vm")
        destroy =0
        create=-10
        seen_actions.append(action)
    return destroy, create, seen_actions


#######################################################################
# REINFORCEMENT LEARNING ENVIRONMENT 
class Application_Env(Env):
    
    def __init__(self,action_value_list,utility):
        '''
        Actions would be:
        0: change the grouping combination
        1: stay as it is
        '''
        self.action_space = Discrete(len(action_value_list)) # number of possible actions
        self.observation_space = Box(low = np.array([0.1]), high = np.array([0.99])) #utility function value boundaries
        '''
        This is the str with all the possible properties for component grouping
        i.e. {'0': [[1, 2, 3]], '1': [[1], [2, 3]], '2': [[3], [1, 2]], '3': [[1], [2], [3]]}
        '''
        self.all_state = action_value_list
        self.list_with_all_possible_actions = action_value_list[:-1]
        self.initial_property = action_value_list[-1]
        self.utility = utility
        self.list_of_utilities = [float(self.utility)]
        self.seen_actions = [self.initial_property]


    def step(self, action): 
        print("Agent tests the {} action".format(action))
        properties = action
        self.utility = trigger_utility()
        
        #print("The agent selects to group components as {} and the utility function is: {}".format(action, self.utility))
        
        destroy, create, self.seen_actions = cost_calculation(action, self.seen_actions)
        '''
        if destroy>create:
            print("The action has as a result to destroy a vm")
        elif destroy == create:
            print("The action has no consequence")
        else:
            print("The action has as a result to create additional vm")    
        '''
        
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

#######################################################################

if __name__ == "__main__":
    time.sleep(1)
    all_combinations = tests.set_of_possible_actions()
    time.sleep(2)
    report_of_app_components(all_combinations)
    action_name_list, action_value_list, action_dictionary = create_the_action_plan(all_combinations) #useful variables for the program
    int_act = initial_state(all_combinations)
    num_of_actions = 2
    utility = trigger_utility() #is the utility for every running component in seperate vms


    
    my_env = Application_Env(action_value_list=action_value_list,utility=utility)

episodes=30

for episode in range(episodes+1):
    initial_utility = my_env.reset()
    done =False
    score=0
    #counter_=0
    c_ = 0
    seen_actions = my_env.reset_seen_actions()
    #print("seen actions {}".format(seen_actions))
    lr = random.sample(my_env.list_with_all_possible_actions, len(my_env.list_with_all_possible_actions))
    while not done and c_ <len(my_env.list_with_all_possible_actions):
        action=lr[c_]
        utility, reward, done, info = my_env.step(action)
        print(reward)
        print(utility)
        #time.sleep(2)
        score+=reward
        c_+=1
        time.sleep(2)
        print("***********************")
    
    print("episode: {}, score: {}".format(episode, score))
    #print(seen_actions)
    print("------------")
    




    

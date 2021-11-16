import numpy as np
from numpy.lib.function_base import diff
import networkx as nx 

"""

Setup variables:

    component_boundary: number of application components (i.e. an application with 3 components indicates that
    the component_boundary has the value 3)

    actions: array-like with the possible actions for application with 2,3,4,5 components (2 components-> 2actions, 
    3 components -> 7 actions, 4 components -> 16 actions etc. )

    states: array-like with the possible states of the environment according to the number of components:

    counter_steps, counter_actions = internal variables that will be used in the next functions in component boundary 
    is greater than 3

"""

components = np.arange(2,6)
states = np.array([3,8,17,30])
actions=np.array([2,7,16,29])
counter_steps=2
counter_actions=3
component_boundary = int(input("Define the number of application components:"))

def create_step_function():

    steps = np.diff(actions)
    for component in range(2,component_boundary+1):
        counter = 2
        new= steps[component]+4
        steps = np.append(steps, new)
        counter = counter+1
    return steps

def create_actions(actions, steps):

    """
    Args: 
        actions: custom-based actions that have been defined in row:7
        steps: the result from  function create_step_function, which produces the number of steps up to the last component
    
    Returns:
        The number of possible actions according to the defined number of components
    """
    counter = 3
    for c_ in range(4, component_boundary+1):
        actions = np.append(actions, steps[counter]+actions[counter])
        counter+=1
    return actions

def create_states(actions,states, component_boundary):
    
    """
    Args:
        actions:  custom-based actions that have been defined in the beggining of the script ( check row: 7)
        state: custom-based states that have been defined similarly in the beggining of the script (check ow: 6)
        component_boundary: this variable is assigned from the user, defines the number of application components
    
    Returns:
        The number of states according to the number of define components.
    """ 

    for st_ in range(4,component_boundary+1):
        states=np.append(states,actions[st_]+1)
    
    return states


if component_boundary >3:
    """
    If the number of components exceeds No. 3 then execute the above functions  
    and return to the user the number of states and components

    """ 
    steps = create_step_function()
    actions = create_actions(actions,steps)
    states=create_states(actions,states,component_boundary)
    print("the number of actions is: {}".format(actions[component_boundary]))
    print("the state-space number is: {}".format(states[component_boundary]))
else:
    """

    Otherwise, return actions, states based custom variables that have been defined in 
    initial rows of this script

    """
    print("the number of actions is: {}".format(actions[component_boundary]))
    print("the state-space number is: {}".format(states[component_boundary]))

    









    


from os import X_OK
from numpy.lib.function_base import diff
import matplotlib.pyplot as plt
import numpy as np
import itertools
import app_components
import collections
import combinations

#SET THE NUMBER OF SUPPORTING VARIABLES 
result=[]
idxs_=[]


def create_sequential_component_list(component_no:int, stuff=[]):
    
    """
    This function creates the sequential list of components

    Args:
        component_no: Number of components
    
    Returns:
        stuff: the list of sequential components
    """
    for c_ in range(1, component_no+1):
        stuff.append(c_)
    return stuff

def create_all_combinations(components):

    """
    This function creates all the possible combination of components

    Args:
        components: the number of components
    
    Returns:
        The list with all the possible combination of the components
    """
    for L in range(0, len(components)+1):
        for subset in itertools.combinations(components, L):
            result.append(list(subset))
    return result
    
def set_rules(result, result_=[]):

    """
    This function sets rules to the possible actions

    Args:
        result: the number of components as a list
    
    Returns:

        the new combination of components based on rules

    """
    for element in result:
        if len(element) == 0:
            idx=result.index(element)
            result.pop(idx)
        elif len(element) ==1:
            pass
        elif len(element)>=2:
            idx_ = result.index(element)
            max_ = max(np.diff(result[idx_]))
            if max_ > 1 :
                idx_=result.index(element)
                idxs_.append(idx_)

    for index_ in idxs_:
        result_.append(result[index_])

    for element in result_:
        result.remove(element)

    return(result)


def create_actions(lst):
    """
    This function finds unique elements in a list 

    Args:
        lst: the provided list
    
    Returns:
        The max frequency value from all elements in the list
    """
    flat_list = [item for sublist in lst for item in sublist]
    counter=collections.Counter(flat_list)
    limit = max(counter.values())
    return limit


def build_actions(lst):

    """
    This function finds all the possible actions that exist in the RL-environment

    Args:
        lst: the list with all the combinations-components based on rules

    Returns: 
        All the opssible actions in the environment
    """
    final=[]
    for element in lst:
        if len(element) ==0:
            lst.remove(element)
        else:
            conc = np.concatenate(element)
        #print(conc)
            sum_ = sum(conc)  
            if sum_ == 6:
                limit = create_actions(element)
                if limit ==1:
                    final.append(element)
                else:
                    pass
            else:
                pass
    return final

def set_of_possible_actions():
    component_no = app_components.main()
    stuff = create_sequential_component_list(component_no)
    result = create_all_combinations(stuff)
    final = set_rules(result)
    new_combinations = combinations.extended_combinations(final)
    
    return combinations.actions(new_combinations)
    
if __name__ == "__main__":
    
    #the number of all possible combinations 
    set_of_possible_actions()
    

    
    


    



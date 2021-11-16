import numpy as np
import itertools
import collections
import app_components


values = [[1], [2], [3], [1, 2], [2, 3], [1, 2, 3]]

def extended_combinations(values):
    result=[]
    for L in range(0, len(values)+1):
        for subset in itertools.combinations(values, L):
            result.append(list(subset))
    return result


def create_actions(result):
    flat_list = [item for sublist in result for item in sublist]
    counter=collections.Counter(flat_list)
    limit = max(counter.values())
    return limit

final = []

def actions(result):
    """
    This function implements the possible actions that we can take inside an application environment

    Args:
        result: (dtype-list), a list with all the possible combinations of application components
    
    Returns:

    """
    for element in result:
        if len(element) ==0:
            result.remove(element)
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


#print(final)
import time



def cost_calculation(list_with_combinations, counter_, seen_list, list_used=[]):
    creation, destroy = 0, 0
    for element in list_with_combinations:
        print("we are testing element {}".format(element))
        time.sleep(2)
    if len(list_with_combinations[counter_]) < len(seen_list[-1]):
        destroy+=1
        creation +=0
        list_used.append(element)
        print( "destroy: {}".format(destroy))
    elif len(list_with_combinations[counter_]) == len(seen_list[-1]):
        print("we do nothing")
    else:
        creation+=1
        destroy += 0
        list_used.append(element)
        print( "create: {}".format(creation))
    
    counter_+=1

    return creation, destroy, counter_, list_used


list_with_combinations = [[[1], [2, 3]], [[3], [1, 2]], [[1], [2], [3]]]
seen_list = [[[1, 2, 3]]]
counter_ =0
creation, destroy, counter_, list_used= cost_calculation(list_with_combinations=list_with_combinations, seen_list=seen_list, counter_=counter_)

print(creation, destroy, counter_)
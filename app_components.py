from posix import listdir
import numpy as np
import pandas as pd
import random
import json
import os

os.chdir("/Users/pepiparaskevoulakou/Desktop")

def find_the_file(filename):
    for file_ in os.listdir():
        if file_.endswith(filename+".json"):
            result = file_
    return result

def export_data(file_):
    final_path = os.path.join(os.getcwd(), file_)
    with open(final_path) as f:
        data = json.load(f)
    return data

def main():
    filename = str(input('provide the name of the file:'))
    file_ = find_the_file(filename)
    data = export_data(file_)
    Communication = data.get("Communication_components")

    return len(Communication)


if __name__ == "__main__":
    main()
    
    
    

        
    




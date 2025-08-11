import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("kicks_classification.csv")
df = df.loc[0:10]

def compute_svm(df):
    out_list = []
    in_array = df.to_numpy()
    
    for row in in_array:
        temp_list = []
        for column in range(1, len(row), 3):
            smv = np.sqrt(row[column]**2 + row[column+1]**2 + row[column+2]**2)
            temp_list.append(smv)
        out_list.append(temp_list)
    
    return np.array(out_list)

print(compute_svm(df))
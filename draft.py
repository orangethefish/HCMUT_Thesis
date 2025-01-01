import pandas as pd
import numpy as np


df = pd.read_csv('synthetic_data/label_0.csv')
#duplicate the data by adding itself 26 times and save to csv
base = df.copy(deep = True)
for i in range(26):
    base = pd.concat([base, df])    
base.to_csv('synthetic_data/label_0.csv', index=False)
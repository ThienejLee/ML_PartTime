import numpy as np
import pandas as pd 

df = pd.read_csv('UNSW_NB15_testing-set.csv', header = None)

df = df[3]

df.to_csv('column3.csv')
#Data Preprocessing.

#Importing Libraries.
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#Importing Datasets.
dataset = pd.read_csv('data.csv') 
X = dataset.iloc[:, :-1].values     # iloc -> Integer based indexing.
y = dataset.iloc[:, 3].values

#taking care of missing data.
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'np.nan', strategy = 'mean', axis = 0)  
# Imputer(missing_values, strategy, axis) 
# strategy -> (mean/median/most_frequent)
# axis -> (1-columns / 0-rows)
imputer = imputer.fit(X[:, 1:3])    #fit in missing columns ie 2 and 3 -> 1:3(3 upper bound is excluded)
X[:, 1:3] = imputer.transform(X[:, 1:3])    #replace data in missing columns
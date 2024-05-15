# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:16:29 2024

@author: admin
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer

df = sns.load_dataset('penguins')
df.head()

x = df.iloc[:, :-1].values
y = df.iloc[:, -1]


# Handling missing data
df.isnull().sum()
df.dropna(inplace=True)


# Imputing missing data
imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_mean.fit(x[:, 2:])
x[:, 2:] = imputer_mean.transform(x[:, 2:])


# OneHot Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

species = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = species.fit_transform(x)
island = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = island.fit_transform(x)


# Label Encoding
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
imputer_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
sex = imputer_mode.fit(y)
y = sex.transform(y)


# Split the dataset for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)





























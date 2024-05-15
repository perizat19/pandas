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


# Find and Remove Outlier

# Quantile technique
# bill_length_mm columm

import matplotlib.pyplot as plt

plt.hist(df['bill_length_mm'], bins=15)
plt.show()

def plot_boxplot(df, ft):
    df.boxplot(column=[ft])
    plt.grid(False)
    plt.show()


plot_boxplot(df, 'bill_length_mm')


def remove_outliers(df, ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    ls = df.index[(df[ft] < lower_bound) | (df[ft] > upper_bound)]
    return ls

for feature in ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']:
    index_list = []
    index_list.extend(remove_outliers(df, feature))
    
    
index_list


def remove_oulier(df, ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df


print(remove_oulier(df, index_list).shape)


# Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train[:, 6:] = scaler.fit_transform(x_train[:, 6:])
x_test[:, 6:] = scaler.fit_transform(x_test[:, 6:])
print(x_train)
print(x_test)



































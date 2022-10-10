# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:13:22 2022

@author: rupak
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("dataset.csv")
print(data.head())

data.isnull().sum()

data["language"].value_counts()

x = np.array(data["Text"])
y = np.array(data["language"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=42)

model = MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)

user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)
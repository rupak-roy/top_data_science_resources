# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:28:16 2022

@author: rupak
"""
'''using the langdetect package in Python which can detect over 55 different languages within a few lines of code.
'''
from langdetect import detect
text = input("Enter any text in any language: ")
print(detect(text))
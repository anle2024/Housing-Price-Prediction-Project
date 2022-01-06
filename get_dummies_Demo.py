# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:04:16 2019

@author: STEVENBOGAERTS
"""

import pandas as pd

def main():
    d = {'A': pd.Series([1, 2, 5, 4, 2]), 'B': pd.Series([6, 7, 4, 7, 6]), 'C': pd.Series([10, 11, 13, 16, 14])}
    wholeDF = pd.DataFrame(d)
    
    # Suppose:
    # - You want column A unchanged
    # - You want column B to be converted to one-hot encoding
    # - You don't want to use column C at all
    # So:
    
    predictors = ['A', 'B']
    df = wholeDF.loc[:, predictors]
    
    print("df before:", df, '\n', sep='\n')
    
    df = pd.get_dummies(df, columns=['B'])
    
    print("df after:", df, '\n', sep='\n')
    
    # print(df.loc[:, 'B']) # error!
    
    # If I need a list of all predictors, I should update the variable
    predictors = df.columns
    
    print("Predictors:", predictors)
    
    
main()
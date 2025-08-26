"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
from math import log

def log2(x):
    return log(x) / log(2)

def mse(Y: pd.Series) -> float:
    """function to calculate mean squared error manually for regression """
    mean_y = sum(Y)/len(Y)
    tot=0
    for i in Y:
        tot += (i - mean_y) ** 2
    return tot/len(Y)

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    new_df= pd.DataFrame()
    for col in X.columns:
        vals= sorted(X[col].unique())
        for val in vals:
            new_df[str(col)+"-"+ str(val)]= (X[col]==val).astype(int)
    return new_df

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_numeric_dtype(y)

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    freq={}
    for val in Y:
        freq[val]= freq.get(val,0)+1
    ans=0
    for val,cnt in freq.items():
        prob= cnt/len(Y)
        ans -= prob*log2(prob)
    
    return ans

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    freq={}
    for val in Y:
        freq[val] = freq.get(val, 0) + 1
    
    ans=1
    for val, cnt in freq.items():
        prob= cnt/len(Y)
        ans-= prob**2
    
    return ans

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion == "information_gain":
        criterion_val= entropy(Y)
    elif criterion == "gini_index":
        criterion_val= gini_index(Y)
    else:
        criterion_val= mse(Y)

    vals = attr.unique()
    weighted_val = 0
    for val in vals:
        subset = Y[attr == val]
        weight = len(subset) / len(Y)
        if criterion == "information_gain":
            weighted_val += weight * entropy(subset)
        elif criterion == "gini_index":
            weighted_val += weight * gini_index(subset)
        else:
            weighted_val += weight * mse(subset)
    
    return criterion_val- weighted_val


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    features: pd.Series is a list of all the attributes we have to split upon
    return: attribute(in the case of discrete input) and threshold(in the case of real input) to split upon
    """
    best_attr= None
    best_thresh=None
    best_ig = -float('inf')
    
    for f in features:
        if check_ifreal(X[f]):
            values = sorted(X[f].unique())
            thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values) - 1)]
            for t in thresholds:
                attr = X[f].apply(lambda v: f"<= {t}" if v <= t else f"> {t}")
                score = information_gain(y, attr, criterion)
                if score > best_ig:
                    best_ig, best_attr, best_thresh = score, f, t
        else:
            categories = sorted(X[f].unique())
            for c in categories:
                attr = X[f].apply(lambda v: f"== {c}" if v == c else f"!= {c}")
                score = information_gain(y, attr, criterion)
                if score > best_ig:
                    best_ig, best_attr, best_thresh = score, f, c
            
    return best_attr,best_thresh


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if check_ifreal(X[attribute]):
        left_idxs = X[attribute] <= value
        right_idxs = X[attribute] > value
    else:
        left_idxs = X[attribute] == value
        right_idxs = X[attribute] != value
        
    return X[left_idxs], y[left_idxs], X[right_idxs], y[right_idxs]

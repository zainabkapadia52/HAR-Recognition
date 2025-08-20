"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd

def log2(x):
    from math import log
    return log(x) / log(2)

def mse(Y: pd.Series) -> float:

    """function to calculate mean squared error manually for regression """
    mean_y = Y.mean()
    return ((Y - mean_y) ** 2).mean()

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_numeric_dtype(y)

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    probs= Y.value_counts(normalize=True)
    return -(probs * probs.apply(lambda p: log2(p) if p > 0 else 0)).sum()

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    probs = Y.value_counts(normalize=True)
    return 1 - sum(probs ** 2)

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


def split_real(X_col: pd.Series, y: pd.Series, criterion: str):
    # Sort values
    df = pd.DataFrame({"feature": X_col, "target": y}).sort_values(by="feature")
    
    unique_vals = df["feature"].unique()
    if len(unique_vals) == 1:
        return None, -float('inf')
    
    best_ig= -float('inf')
    best_thresh= None
    
    # potential threshold values
    thresholds = [(unique_vals[i] + unique_vals[i+1]) / 2 for i in range(len(unique_vals)-1)]
    for t in thresholds:
        left_y= df[df["feature"] <= t]["target"]
        right_y= df[df["feature"] > t]["target"]
        
        if left_y.empty or right_y.empty:
            continue
        
        if criterion == "information_gain":
            criterion_val = entropy(y)
            left_val, right_val = entropy(left_y), entropy(right_y)
        elif criterion == "gini_index":
            criterion_val = gini_index(y)
            left_val, right_val = gini_index(left_y), gini_index(right_y)
        else:
            criterion_val = mse(y)
            left_val, right_val = mse(left_y), mse(right_y)
       
        weight_left= len(left_y) / len(y)
        weight_right= len(right_y) / len(y)
        
        total_val = weight_left * left_val + weight_right * right_val
        ig = criterion_val - total_val
        
        if ig > best_ig:
            best_ig= ig
            best_thresh= t
    
    return best_thresh, best_ig

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
        if pd.api.types.is_numeric_dtype(X[f]):
            # real valued feature
            thresh, ig = split_real(X[f], y, criterion)
            if ig > best_ig:
                best_ig, best_attr, best_thresh = ig, f, thresh
        else:
            # discrete feature
            ig= information_gain(y, X[f], criterion)
            if ig > best_ig:
                best_ig, best_attr, best_thresh = ig, f, None
            
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
    if pd.api.types.is_numeric_dtype(X[attribute]):
        left_idxs = X[attribute] <= value
        right_idxs = X[attribute] > value
    else:
        left_idxs = X[attribute] == value
        right_idxs = X[attribute] != value
        
    return X[left_idxs], y[left_idxs], X[right_idxs], y[right_idxs]

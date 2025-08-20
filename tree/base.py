"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value 

class DecisionTree:
    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if check_ifreal(y):
            criterion = "mse"
        else:
            criterion = self.criterion
        if depth >= self.max_depth or len(y.unique()) == 1 or X.empty:
            if check_ifreal(y):
                return Node(value=float(y.mean()))
            else:
                return Node(value=y.mode()[0])  
            
        features = X.columns
        best_attr, best_thresh = opt_split_attribute(X, y, criterion, features)

        if best_attr is None: 
            if check_ifreal(y):
                return Node(value= float(y.mean()))
            else:
                return Node(value=y.mode()[0])
        
        X_left, y_left, X_right, y_right = split_data(X, y, best_attr, best_thresh)

        if X_left.empty or X_right.empty:
            if check_ifreal(y):
                return Node(value=float(y.mean()))
            else:
                return Node(value=y.mode()[0])

        left= self._build_tree(X_left, y_left, depth + 1)
        right= self._build_tree(X_right, y_right, depth + 1)

        return Node(feature=best_attr, threshold=best_thresh, left=left, right=right)


    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = [self._traverse_tree(x, self.root) for _, x in X.iterrows()]
        return pd.Series(preds)

    def _traverse_tree(self, x, node: Node):     
        if node.value is not None:
            return node.value
        if pd.api.types.is_numeric_dtype(type(x[node.feature])):
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else: 
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        

    def plot(self, node=None, depth=0):

        if node is None:
            node = self.root
        if node.value is not None:
            print("\t" * depth + f"Leaf: {node.value}")
        else:
            print("\t" * depth + f"{node.feature} <= {node.threshold}")
            self.plot(node.left, depth + 1)
            self.plot(node.right, depth + 1)
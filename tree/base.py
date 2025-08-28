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

@dataclass
class Node:
    def __init__(self):
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.split_attr = None
        self.children = {}  # for discrete splits

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int
    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = {}
        self.inp = None
        self.out = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        feature= [X[i].dtype.name for i in X.columns]

        self.inp= feature[0]
        self.out= y.dtype.name

        if self.inp != "float64":
            X = one_hot_encoding(X)  #one hot categorical inputs if discrete .

        attr = pd.Series(X.columns)

        if self.inp != "float64" and self.out != "float64":
            self.root = self.fit_discrete_discrete(X, y, attr, 0)

        elif self.inp != "float64" and self.out == "float64":
            self.root = self.fit_discrete_real(X, y, attr, 0)

        elif self.inp == "float64" and self.out != "float64":
            self.root = self.fit_real_discrete(X, y, attr, 0)

        elif self.inp == "float64" and self.out == "float64":
            self.root = self.fit_real_real(X, y, attr, 0)

    
    def fit_discrete_discrete(self, X,y, attr,depth):
        if depth >= self.max_depth:
            return y.mode().iloc[0]
        if len(y.unique()) == 1:
            return y.iloc[0]
        if len(attr) == 0:
            return y.mode().iloc[0]

        best_attr = opt_split_attribute(X, y, self.criterion, attr)
        node = Node()
        node.split_attr = best_attr

        encoded_features = [col for col in X.columns if col.startswith(str(best_attr) + "-")]
        attr = attr[~attr.isin(encoded_features)]

        for _, feat in enumerate(encoded_features):
            X_subset = X[X[feat] == 1]
            y_subset = y[X_subset.index]
            if X_subset.empty:
                node.children[feat] = y.mode().iloc[0]
                node.value = y.mode().iloc[0]
            else:
                node.children[feat] = self.fit_discrete_discrete(X_subset, y_subset, attr, depth + 1)
        return node
    
    def fit_discrete_real(self, X, y, attr, depth):
        if depth >= self.max_depth or len(attr) == 0: 
            return y.mean()
        if len(y.unique()) == 1: 
            return y.iloc[0]

        #Group one hot columns by their original feature
        groups = {}
        for col in attr:
            orig = str(col).split("-")[0] 
            groups.setdefault(orig, []).append(col)

        # Pick the feature with the best variance reduction
        best_attr, best_score = None, -float("inf")
        for orig, cols in groups.items():
            active = X[cols].idxmax(axis=1)
            score = information_gain(y, active, "mse")
            if score > best_score:
                best_score, best_attr = score, orig

        node = Node()
        node.split_attr = best_attr
        node.value = y.mean() 
        
        encoded_features = groups[best_attr]
        attr = attr[~attr.isin(encoded_features)]

        for col in encoded_features:
            X_subset = X[X[col] == 1]
            y_subset = y[X_subset.index]
            if X_subset.empty:
                node.children[col] = y.mean()
            else:
                node.children[col] = self.fit_discrete_real(X_subset, y_subset, attr, depth + 1)

        return node

    def fit_real_discrete(self, X, y, attr, depth):
        if depth >= self.max_depth: return y.mode().iloc[0]
        if len(y.unique()) == 1: return y.iloc[0]
        if len(attr) == 0: return y.mode().iloc[0]

        best_attr, best_thresh = opt_split_attribute(X, y, self.criterion, attr)
        node = Node()
        node.split_attr = best_attr
        node.threshold = best_thresh

        attr = attr[~attr.isin([best_attr])]

        X_left, y_left, X_right, y_right = split_data(X, y, best_attr, best_thresh)
        node.left = self.fit_real_discrete(X_left, y_left, attr, depth + 1)
        node.right = self.fit_real_discrete(X_right, y_right, attr, depth + 1)
        return node
    
    def fit_real_real(self, X, y, attr, depth):
        if depth >= self.max_depth or len(attr) == 0: return y.mean()
        if len(y.unique()) == 1: return y.iloc[0]

        best_attr, best_thresh = opt_split_attribute(X, y, "mse", attr)
        node = Node()
        node.split_attr = best_attr
        node.threshold = best_thresh

        attr = attr[~attr.isin([best_attr])]

        X_left, y_left, X_right, y_right = split_data(X, y, best_attr, best_thresh)
        node.left = self.fit_real_real(X_left, y_left, attr, depth + 1)
        node.right = self.fit_real_real(X_right, y_right, attr, depth + 1)
        return node
   
    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.inp != "float64":
            X = one_hot_encoding(X)

        preds = []
        for _, row in X.iterrows():
            node = self.root
            while isinstance(node, Node):
                split_attr, threshold = node.split_attr, node.threshold
                if self.inp == "float64":  # real input
                    if row[split_attr] <= threshold:
                        node = node.left
                    else:
                        node = node.right
                else:  # discrete input
                    found = False
                    for feat, child in node.children.items():
                        if row[feat] == 1:  
                            node = child
                            found = True
                            break
                    if not found:
                        node = node.value
            preds.append(node)
        return pd.Series(preds)

    def plot(self, node=None, indent=""):
        if node is None:
            node = self.root

        # Leaf node
        if not isinstance(node, Node):
            print(indent + f"Prediction: {node}")
            return

        # Real-valued splits
        if self.inp == "float64":
            print(indent + f"? (X[{node.split_attr}] <= {node.threshold:.4f})")

            print(indent + "  True:")
            self.plot(node.left, indent + "    ")

            print(indent + "  False:")
            self.plot(node.right, indent + "    ")

        # Discrete splits
        else:
            print(indent + f"? (X[{node.split_attr}] categorical split)")
            for value, child in node.children.items():
                print(indent + f"  Value = {value}:")
                self.plot(child, indent + "    ")


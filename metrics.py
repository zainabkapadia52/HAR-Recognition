from typing import Union
import pandas as pd
import math

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    """
    assert y_hat.size == y.size
    y_hat = list(y_hat)
    y = list(y)

    correct = 0
    for i in range(len(y)):
        if y_hat[i] == y[i]:
            correct += 1

    return correct / len(y)



def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    y_hat = list(y_hat)
    y = list(y)

    tp = 0  # true positives
    fp = 0  # false positives

    for i in range(len(y)):
        if y_hat[i] == cls:  # predicted positive
            if y[i] == cls:
                tp += 1
            else:
                fp += 1
    if(tp+fp)==0:
        return 0.0
    
    return tp / (tp + fp)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    y_hat = list(y_hat)
    y = list(y)

    tp = 0  # true positives
    fn = 0  # false negatives

    for i in range(len(y)):
        if y[i] == cls:  # actual positive
            if y_hat[i] == cls:
                tp += 1
            else:
                fn += 1
    if(tp+fn)==0:
        return 0.0
    
    return tp/(tp+fn)


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    y_hat = list(y_hat)
    y = list(y)

    err = []
    for i in range(len(y)):
        err.append((y_hat[i] - y[i]) ** 2)

    return (sum(err) / len(y))**0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    y_hat = list(y_hat)
    y = list(y)

    abs_err= []
    for i in range(len(y)):
        abs_err.append(abs(y_hat[i] - y[i]))

    return sum(abs_err) / len(y)

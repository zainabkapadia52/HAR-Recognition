from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    """
    assert y_hat.size == y.size
    return (pd.Series(list(y_hat)) == pd.Series(list(y))).sum() / len(y)



def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    y_hat = pd.Series(list(y_hat))
    y = pd.Series(list(y))
    assert y_hat.size == y.size
    tp= ((y_hat == cls) & (y == cls)).sum()
    fp= ((y_hat == cls) & (y != cls)).sum()
    if(tp+fp)>0:
        return tp / (tp + fp)
    else:
        return 0.0


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    y_hat = pd.Series(list(y_hat))
    y = pd.Series(list(y))
    tp= ((y_hat == cls) & (y == cls)).sum()
    fn= ((y_hat != cls) & (y == cls)).sum()
    if(tp+fn)>0:
        return tp/(tp+fn)
    else:
        return 0.0


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    return ((y_hat - y) ** 2).mean() ** 0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    return (y_hat - y).abs().mean()

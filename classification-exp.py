import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from metrics import accuracy, precision, recall
from tree.base import DecisionTree


np.random.seed(42)

# # Write the code for Q2 a) and b) below. Show your results.
#Part a
# Code given in the question
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
X_df = pd.DataFrame(X, columns=['Feat1','Feat2'])
y_series = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.3, random_state=42, shuffle=True)
tree = DecisionTree(criterion="gini_index")
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("Accuracy:", accuracy(y_pred, y_test))
for cls in y_test.unique():
    print(f"precision of class {cls}: ", precision(y_pred, y_test, cls))
    print(f"recall of class {cls}: ", recall(y_pred, y_test, cls))

#Part b
    
from sklearn.model_selection import KFold
max_depths= [1,2,3,4,5,6,7,8,9,10]
num_outer_folds= 5
num_inner_folds= 5

kf_outer= KFold(n_splits=num_outer_folds, shuffle=True, random_state=42)
kf_inner= KFold(n_splits=num_inner_folds, shuffle=True, random_state=42)

outer_loop_acc= []
inner_loop_acc= []

depth_val_scores_all= {d: [] for d in max_depths}
res=[]
final_depths=[]

#outer loop
for i, (train_val_idx, test_idx) in enumerate(kf_outer.split(X)):
    X_train_val, X_test = X_df.iloc[train_val_idx], X_df.iloc[test_idx]
    y_train_val, y_test = y_series.iloc[train_val_idx], y_series.iloc[test_idx]

    depth_val_scores= {d: [] for d in max_depths}

    # inner loop for hyperparameter tuning
    for inner_train_idx, inner_val_idx in kf_inner.split(X_train_val):
        X_inner_train, X_inner_val = X_train_val.iloc[inner_train_idx], X_train_val.iloc[inner_val_idx]
        y_inner_train, y_inner_val = y_train_val.iloc[inner_train_idx], y_train_val.iloc[inner_val_idx]

        for d in max_depths:
            clf= DecisionTree(criterion="gini", max_depth=d)
            clf.fit(X_inner_train, y_inner_train)
            val_pred= clf.predict(X_inner_val)
            acc= accuracy(y_inner_val, val_pred)
            depth_val_scores[d].append(acc)
            depth_val_scores_all[d].append(acc)

    # Average inner fold performance per depth
    mean_val_scores= {d: np.mean(scores) for d, scores in depth_val_scores.items()}

    # depth with best mean accuracy
    best_depth = max(mean_val_scores, key=mean_val_scores.get)
    final_depths.append(best_depth)

    final_clf= DecisionTree(criterion="gini", max_depth=best_depth)
    final_clf.fit(X_train_val, y_train_val)
    outer_pred= final_clf.predict(X_test)
    outer_acc= accuracy(y_test, outer_pred)
    res.append(outer_acc)

    print(f"Outer Fold {i+1}: Best depth = {best_depth}, Test accuracy = {outer_acc:.4f}")

print("\nNested CV Results")
print(f"Chosen depths across folds: {final_depths}")
print(f"Mean Test Accuracy: {np.mean(res):.4f}")
print(f"Std Test Accuracy: {np.std(res):.4f}")

mean_depth_scores= {d: np.mean(scores) for d, scores in depth_val_scores_all.items()}
optimum_depth = max(mean_depth_scores, key=mean_depth_scores.get)

print(f"\nOptimum depth of the tree (based on avg inner validation accuracy): {optimum_depth}")

final_model = DecisionTree(criterion="gini_index", max_depth=optimum_depth)
final_model.fit(X_df, y_series)
y_full_pred = final_model.predict(X_df)
final_acc = accuracy(y_full_pred, y_series)

print(f"\nFinal model trained on full dataset with depth={optimum_depth}")
print(f"Overall Accuracy on full dataset: {final_acc:.4f}")
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from pprint import pprint
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import mltools as ml

X = np.loadtxt("data/X_train.txt")
Y = np.loadtxt("data/Y_train.txt")

## shuffle and split
# Xtr,Ytr = ml.shuffleData(X,Y)
Xtr = X[:10000]
Ytr = Y[:10000]
Xtr,Xva,Ytr,Yva = ml.splitData(Xtr,Ytr,0.25)

print(Xtr.shape)
print(Xva.shape)

# Parameters
n_estimators = [1000,1500,2000]
max_features = ['auto', 'log2']
max_depth = [1,4,7]
learning_rate = [1,.1,0.01,0.001]
min_samples_split = [2, 6, 10]
min_samples_leaf = [2, 5, 10]

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
# params = {
#                'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'learning_rate': learning_rate,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
# }

# estimator = GradientBoostingClassifier(random_state = 42)
# estimator.get_params()
# randomized = RandomizedSearchCV(estimator = estimator, param_distributions = params, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# # Fit the random search model
# randomized.fit(Xtr, Ytr)
# print("Best parameters:", randomized.best_params_)


## GRID SEARCH ####
# Create the parameter grid based on the results of random search 
params = {
               'n_estimators': [1250],
               'max_features': ['auto'],
               'max_depth': [4,5,6],
               'learning_rate': [0.001, 0.002,0.003],
               'min_samples_split': [2,3,4],
               'min_samples_leaf': [5,6,7],
}

# Create a based model
# Instantiate the grid search model
estimator = GradientBoostingClassifier(random_state = 42)
grid_search = GridSearchCV(estimator = estimator, param_grid = params, cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(Xtr, Ytr)
print("Best parameters:", grid_search.best_params_)
# gb = GradientBoostingClassifier(n_estimators=1250, learning_rate = .005, max_features="auto", max_depth = 1, min_samples_leaf= 11, min_samples_split= 5)
# gb.fit(Xtr, Ytr)
# print("Accuracy score (training): {0:.3f}".format(gb.score(Xtr, Ytr)))
# print("Accuracy score (validation): {0:.3f}".format(gb.score(Xva, Yva)))
# print()

# gb = GradientBoostingClassifier(n_estimators=1250, learning_rate = .001, max_features="auto", max_depth = 4, min_samples_leaf= 5, min_samples_split= 2)
# gb.fit(Xtr, Ytr)
# print("Accuracy score (training): {0:.3f}".format(gb.score(Xtr, Ytr)))
# print("Accuracy score (validation): {0:.3f}".format(gb.score(Xva, Yva)))
# print()
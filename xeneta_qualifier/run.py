import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def convertToFloat(lst):
    return np.array(lst).astype(np.float)

def fetchData(path):
    labels = []
    data = []
    f = open(path)
    csv_f = csv.reader(f)
    for row in csv_f:
        labels.append(convertToFloat(row[0]))
        data.append(convertToFloat(row[1:]))
    f.close()
    return np.array(data), np.array(labels)

# Random Forest Classifier
def runForest(X_train, y_train):
    forest = RandomForestClassifier(n_estimators=90, random_state=42)
    forest.fit(X_train, y_train)
    return forest

# Stochastic Gradient Descent Classifier
def runSGD(X_train, y_train):
    sgd = SGDClassifier(max_iter=500, loss='modified_huber', penalty='elasticnet', random_state=42)
    sgd.fit(X_train, y_train)
    return sgd
    
# Gradient Boosting Classifier
def runGBC(X_train, y_train):
  boost = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
  boost.fit(X_train, y_train)
  return boost

# Ada Boost Classifier
def runAda(X_train, y_train):
  ada = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
  ada.fit(X_train, y_train)
  return ada
 
 # Bagging Classifier
def runBag(X_train, y_train):
  bag = BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
  bag.fit(X_train, y_train)
  return bag
 
 # Scores
def getScores(clf, X, y):
  predictions = clf.predict(X)
  scores = precision_recall_fscore_support(y, predictions, average='binary')
  return scores

# Import data
X_test, y_test = fetchData('data/test.csv')
X_train, y_train = fetchData('data/train.csv')

# Run classifiers
forest = runForest(X_train, y_train)
forest_scores = getScores(forest, X_test, y_test)
print('Random Forest Scores: ', forest_scores)

sgd = runSGD(X_train, y_train)
sgd_scores = getScores(sgd, X_test, y_test)
print('SGD Scores: ', sgd_scores)

boost = runGBC(X_train, y_train)
boost_scores = getScores(boost, X_test, y_test)
print('Gradient Boosting Scores: ', boost_scores)

ada = runAda(X_train, y_train)
ada_scores = getScores(ada, X_test, y_test)
print('Ada Boost Scores: ', ada_scores)

bag = runBag(X_train, y_train)
bag_scores = getScores(bag, X_test, y_test)
print('Bagging Classifier Scores: ', bag_scores)
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
def get_train_data(filename):
    X  = []
    y_task1 = []
    y_task2 = []
    
    with open(filename, encoding='UTF-8') as file:
        for line in file:
            tweet = line.rstrip('\n').split('\t')
            X.append(tweet[0])
            y_task1.append(tweet[1])
            y_task2.append(tweet[2])
    
    return np.asarray(X), np.asarray(y_task1), np.asarray(y_task2)

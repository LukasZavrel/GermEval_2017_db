#%%
import numpy as np
import pandas as pd
import os
#%%
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
#%%
data_train = pd.read_csv("./GermEval 2017 Shared Task/data/train_v1.4.tsv", sep="\t", header=None)
data_valid = pd.read_csv("./GermEval 2017 Shared Task/data/dev_v1.4.tsv", sep="\t", header=None)


#%%
no_tweets = list(data_train[data_train.iloc[:,1].apply(lambda x: not isinstance(x, str))].index)
remaining_tweets = [index for index in list(data_train.index) if index not in no_tweets]
data_train = data_train.loc[remaining_tweets, :]
#%%
no_tweets = list(data_valid[data_valid.iloc[:,1].apply(lambda x: not isinstance(x, str))].index)
remaining_tweets = [index for index in list(data_valid.index) if index not in no_tweets]
data_valid = data_valid.loc[remaining_tweets, :]
#%%
X_train = data_train.iloc[:,1]
Y_train_1 = data_train.iloc[:,2]
Y_train_2 = data_train.iloc[:,3]
X_valid = data_valid.iloc[:,1]
Y_valid_1 = data_valid.iloc[:,2]
Y_valid_2 = data_valid.iloc[:,3]

#%%
count_vect = CountVectorizer(min_df=1)
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape

#%%
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

#%%
clf = MultinomialNB().fit(X_train_tfidf, Y_train_1)

#%%
predicted=clf.predict(tfidf_transformer.transform(count_vect.transform(X_valid)))
np.mean(predicted == Y_valid_1)
# %%
from sklearn.metrics import f1_score
f1_score(predicted, Y_valid_1, average='macro')

#%%
f1_score(predicted, Y_valid_1, average=None)

#%%
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
# %%
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(text_clf, parameters, cv=10, iid=False,
                      n_jobs=-1, scoring=make_scorer(f1_score, average='macro'))
gs_clf = gs_clf.fit(X_train, Y_train_1)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

#%%
predicted=gs_clf.predict(X_valid)
np.mean(predicted == Y_valid_1)

#%%
f1_score(predicted, Y_valid_1, average='macro')

#%%
f1_score(predicted, Y_valid_1, average=None)

#%%


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
SVM.fit(X_train_tfidf, Y_train_1)

predicted = SVM.predict(tfidf_transformer.transform(count_vect.transform(X_valid)))
np.mean(predicted == Y_valid_1)

#%%
f1_score(predicted, Y_valid_1, average='micro')

#%%
f1_score(predicted, Y_valid_1, average=None)

#%%
naive_bayes_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
parameters = {
    'vect__ngram_range': (1, 2),
    'tfidf__use_idf': True,
    'clf__alpha': 1e-2,
}
naive_bayes_clf.set_params(**parameters)

#%%
naive_bayes_clf.fit(X_train, Y_train_1)
#%%
predicted=naive_bayes_clf.predict(X_valid)
np.mean(predicted == Y_valid_1)

#%%
svm_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)),
])
svm_clf.fit(X_train, Y_train_1)

#%%
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict_proba(inputX)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = np.dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX
#%%
def fit_stacked_model(members, inputX, inputy):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# fit standalone model
	model = LogisticRegression()
	model.fit(stackedX, inputy)
	return model
#%%
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat
#%%
stacked_model = fit_stacked_model([naive_bayes_clf, svm_clf], X_train, Y_train_1)


#%%
predicted = stacked_prediction([naive_bayes_clf, svm_clf], stacked_model, X_valid)

#%%
f1_score(predicted, Y_valid_1, average='micro')

#%%
f1_score(predicted, Y_valid_1, average=None)

#%%

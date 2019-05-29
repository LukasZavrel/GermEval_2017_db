# %%
from collections import Counter
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from scripts.utilities import *
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
filename = "./data/germeval2018.training.txt"
X_train, Y_train1, Y_train2 = get_train_data(filename)
filename = "./data/germeval2018.test.txt"
X_test, Y_test1, Y_test2 = get_train_data(filename)

# %%
count_vect = CountVectorizer(min_df=1)
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape

# %%
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# %%
Y_train1[:10]

# %%
clf = MultinomialNB().fit(X_train_tfidf, Y_train1)
# %%
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
gs_clf = gs_clf.fit(X_train, Y_train1)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


# %%
gs_clf.best_score_
# %%
Counter(Y_test1)
print("Default score: ", 2330/(2330+1202))

# %%
text_clf.set_params(**gs_clf.best_params_)
text_clf.fit(X_train, Y_train1)
# %%
predicted=text_clf.predict(X_test)
np.mean(predicted == Y_test1)
# %%
from sklearn.metrics import f1_score
f1_score(predicted, Y_test1, average='macro')

# %%
from nltk.tokenize import TweetTokenizer as Tokenizer_NLTK
from nltk.tokenize.casual import remove_handles
from nltk.stem.snowball import GermanStemmer as Stemmer_NLTK
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
class Tokenizer:
    def __init__(self, preserve_case=True, use_stemmer=False, join=False):
        self.preserve_case=preserve_case
        self.use_stemmer=use_stemmer
        self.join=join

    def tokenize(self, tweet):
        tweet=remove_handles(tweet)
        tweet=tweet.replace('#', ' ')
        tweet=tweet.replace('&lt;', ' ')
        tweet=tweet.replace('&gt;', ' ')
        tweet=tweet.replace('&amp;', ' und ')
        tweet=tweet.replace('|LBR|', ' ')
        tweet=tweet.replace('-', ' ')
        tweet=tweet.replace('_', ' ')
        tweet=tweet.replace("'s", ' ')
        tweet=tweet.replace(",", ' ')
        tweet=tweet.replace(";", ' ')
        tweet=tweet.replace(":", ' ')
        tweet=tweet.replace("/", ' ')
        tweet=tweet.replace("+", ' ')
        tknzr=Tokenizer_NLTK(preserve_case=self.preserve_case, reduce_len=True)

        if self.join:
            return " ".join(tknzr.tokenize(tweet))
        elif self.use_stemmer:
            stmmr=Stemmer_NLTK()
            return [stmmr.stem(token) for token in tknzr.tokenize(tweet)]
        else:
            return tknzr.tokenize(tweet)
# %%
token_vect=TfidfVectorizer(analyzer="word", max_df=0.01, min_df=0.0002,
                             tokenizer=Tokenizer(preserve_case=False, use_stemmer=True).tokenize)

char_vect  = TfidfVectorizer(analyzer="char", ngram_range=(3, 7), max_df=0.01, min_df=0.0002,
                             preprocessor=Tokenizer(preserve_case=False, join=True).tokenize)
# %%
X_TNGR_train = token_vect.fit_transform(X_train)
X_TNGR_test  = token_vect.transform(X_test)

X_CNGR_train = char_vect.fit_transform(X_train)
X_CNGR_test  = char_vect.transform(X_test)
#%%

#%%
text_clf = Pipeline([
    ('vect', token_vect),
    ('clf', MultinomialNB()),
])
parameters={
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
    'vect__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}
gs_clf=GridSearchCV(text_clf, parameters, cv=StratifiedKFold(n_splits=10), iid=False,
                      n_jobs=-1, scoring=make_scorer(f1_score, average='macro'))
gs_clf=gs_clf.fit(X_train, Y_train1)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# %%
gs_clf.best_score_

# %%
text_clf.set_params(**gs_clf.best_params_)
text_clf.fit(X_train, Y_train1)
predicted=text_clf.predict(X_test)
np.mean(predicted == Y_test1)
# %%
from sklearn.metrics import f1_score
f1_score(predicted, Y_test1, average='macro')
# %%
token_vect=TfidfVectorizer(analyzer="word", max_df=0.01, min_df=0.0002,
                             tokenizer=Tokenizer(preserve_case=False, use_stemmer=True).tokenize)

char_vect  = TfidfVectorizer(analyzer="char", ngram_range=(3, 7), max_df=0.01, min_df=0.0002,
                             preprocessor=Tokenizer(preserve_case=False, join=True).tokenize)
# %%

X_TNGR_train = token_vect.fit_transform(X_train)
X_TNGR_test  = token_vect.transform(X_test)

X_CNGR_train = char_vect.fit_transform(X_train)
X_CNGR_test  = char_vect.transform(X_test)

#%%
def get_META_feats(clf, X_train, X_test, y, seeds=[42]):
    feats_train = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        feats_train.append(cross_val_predict(clf, X_train, y=y, method='predict_proba', cv=skf, n_jobs=-1))
    feats_train = np.mean(feats_train, axis=0)
    
    clf.fit(X_train, y)
    feats_test = clf.predict_proba(X_test)
    
    return feats_train, feats_test

#%%
clfs_task1 = [LogisticRegression(class_weight='balanced'),
              ExtraTreesClassifier(n_estimators=100, criterion='entropy', n_jobs=-1),
              ExtraTreesClassifier(n_estimators=100, criterion='gini', n_jobs=-1)]

base_feats_task1 = [#(X_CNGR_train, X_CNGR_test),
                    (X_TNGR_train, X_TNGR_test)]

X_META_task1_train = []
X_META_task1_test  = []
for X_train, X_test in base_feats_task1:
    for clf in clfs_task1:
        feats = get_META_feats(clf, X_train, X_test, Y_train1)
        X_META_task1_train.append(feats[0])
        X_META_task1_test.append(feats[1])
        
X_META_task1_train = np.concatenate(X_META_task1_train, axis=1)
X_META_task1_test  = np.concatenate(X_META_task1_test, axis=1)


#%%
clf_task1 = LogisticRegression(C=0.17, class_weight='balanced')
clf_task1.fit(X_META_task1_train, Y_train1)

preds_task1 = clf_task1.predict(X_META_task1_test)    


#%%
np.mean(preds_task1 == Y_test1)
# %%
from sklearn.metrics import f1_score
f1_score(preds_task1, Y_test1, average='macro')

#%%

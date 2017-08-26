
# coding: utf-8

# In[1]:

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents_f = open("documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features_f = open("word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

word_features_f = open("feature_set.pickle", "rb")
feature_set = pickle.load(word_features_f)
word_features_f.close()

classifier_f=open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f=open("MNB_classifier.pickle","rb")
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f=open("BernoulliNB_classifier.pickle","rb")
BernoulliNB_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f=open("LogisticRegression_classifier.pickle","rb")
LogisticRegression_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f=open("SGDClassifier_classifier.pickle","rb")
SGDClassifier_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f=open("SVC_classifier.pickle","rb")
SVC_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f=open("LinearSVC_classifier.pickle","rb")
LinearSVC_classifier = pickle.load(classifier_f)
classifier_f.close()

voted_classifier = VoteClassifier(classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,
                                 SGDClassifier_classifier,SVC_classifier,LinearSVC_classifier)
def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats)


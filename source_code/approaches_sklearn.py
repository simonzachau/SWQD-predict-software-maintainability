import numpy as np
import pandas as pd
import re
from utils import split_camel_case, flatten_list, read_image
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from utils import print_section


class SklearnApproach:
	def __init__(self, name, classifier_pipeline, grid_params=None):
		self.name = name

		# GPU/CPU
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.has_grid_params = grid_params != None

		if grid_params != None:
			self.classifier = GridSearchCV(Pipeline(classifier_pipeline), grid_params)
		else:
			self.classifier = Pipeline(classifier_pipeline)
		self.model = None


	def train(self, x, y):
		print_section('train model: ' + self.name)

		self.model = self.classifier.fit(x, y)

		if self.has_grid_params:
			return self.classifier.best_params_

		return None


	def test(self, x):
		print_section('test model: ' + self.name)

		return self.model.predict(x)


	def build_params(self, param_names):
		all_available_params = {
			'SVM__kernel': ['poly', 'rbf', 'sigmoid'],
			'vect__ngram_range': [(1, 1), (2, 2), (3, 3), (1, 2), (2, 3), (1, 3)],
			'vect__activate_camel_case_split': [True, False],
			'vect__activate_stemming': [True, False]
		}

		selected_params = {key: all_available_params[key] for key in param_names}

		return selected_params


class NaiveBayesTextBased(SklearnApproach):
	def __init__(self):
		title = 'Naives Bayes Text-Based'
		pipeline = [
			('vect', TextVectorizer()),
			('tfidf', TfidfTransformer()),
			('clf', MultinomialNB())]
		params = super().build_params([
			'vect__ngram_range',
			'vect__activate_camel_case_split',
			'vect__activate_stemming'])
		super().__init__(title, pipeline, params)


class SVMTextBased(SklearnApproach):
	def __init__(self):
		title = 'Support Vector Machine Text-Based'
		pipeline = [
			('vect', TextVectorizer()),
			('tfidf', TfidfTransformer()),
			('SVM', SVC())]
		params = super().build_params([
			'vect__ngram_range',
			'SVM__kernel',
			'vect__activate_camel_case_split',
			'vect__activate_stemming'])
		super().__init__(title, pipeline, params)


class SVMImageBased(SklearnApproach):
	def __init__(self):
		title = 'Support Vector Machine Image-Based'
		pipeline = [
			('vect', ImageVectorizer()),
			('SVM', SVC())]
		params = super().build_params(['SVM__kernel'])
		super().__init__(title, pipeline, params)


class TextVectorizer(CountVectorizer):
	def __init__(self, ngram_range=(1, 1), activate_camel_case_split=False, activate_stemming=False):
		self.activate_camel_case_split = activate_camel_case_split
		self.activate_stemming = activate_stemming
		super().__init__(preprocessor=self.preprocessor, ngram_range=ngram_range)


	def preprocessor(self, text):
		# remove special chars and make list of words
		words = re.split('\\s+', re.sub('\\W', ' ', text))

		if self.activate_camel_case_split:
			words = flatten_list(map(lambda word: split_camel_case(word), words))

		if self.activate_stemming:
			stemmer = SnowballStemmer('english', ignore_stopwords=True)
			words = [stemmer.stem(word) for word in words]

		return ' '.join(words).lower()


class ImageVectorizer(BaseEstimator, TransformerMixin):
	# takes an image path and returns a vector
	def __init__(self):
		super().__init__()

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		return list(map(lambda image_path: read_image(image_path).flatten(), X))

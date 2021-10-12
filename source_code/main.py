import os
import copy
import numpy as np
import nltk
from dataset import Dataset
from approaches_bert import BERT, RoBERTa, CodeBERT
from approaches_sklearn import NaiveBayesTextBased, SVMTextBased, SVMImageBased
from approaches_cnn import AlexNet
from evaluation import evaluate
from utils import print_section, write_log_file, reset_seed


def run_approaches_bert(dataset, dimension, approach, frac_training, frac_test, frac_validation=0):
	dataset_input_column_name = 'bert_ids_' + approach.name

	# every approach needs a copy because it changes the dataframe, which is not wanted for the other approaches
	dataset_approach = copy.deepcopy(dataset)

	dataset_approach.bert_tokenize(approach.tokenizer, dataset_input_column_name)
	
	datasets = dataset_approach.stratify_split(frac_training, frac_validation, frac_test, dimension)

	if frac_validation > 0:
		# prepare x
		input_training, input_validation, input_testing = \
		map(lambda dataset: np.vstack(dataset[dataset_input_column_name].values).astype(np.int), datasets)
		
		# prepare y
		labels_training, labels_validation, labels_testing = \
		map(lambda dataset: dataset[dimension].values, datasets)

		approach.train(input_training, labels_training, input_validation, labels_validation)

	else:
		# prepare x
		input_training, input_testing = \
		map(lambda dataset: np.vstack(dataset[dataset_input_column_name].values).astype(np.int), datasets)
		
		# prepare y
		labels_training, labels_testing = \
		map(lambda dataset: dataset[dimension].values, datasets)

		approach.train(input_training, labels_training)

	results = evaluate(approach.test(input_testing), labels_testing)
	statistic_training = ['training distribution:', datasets[0][dimension].value_counts()]
	statistic_testing = ['testing distribution:', datasets[len(datasets) - 1][dimension].value_counts()]
	
	logs = results + statistic_training + statistic_testing
	write_log_file(dataset.name, dimension, approach.name, logs)


def run_approaches_other(dataset, dimension, approach, dataset_input_column_name, frac_training, frac_test):
	dataset_training, dataset_test = dataset.stratify_split(frac_training, 0.0, frac_test, dimension)

	# prepare x
	input_training = dataset_training[dataset_input_column_name]
	input_testing = dataset_test[dataset_input_column_name]

	# prepare y
	labels_training = dataset_training[dimension]
	labels_testing = dataset_test[dimension].values

	gridsearch_cv_params = approach.train(input_training, labels_training)
	
	results = evaluate(approach.test(input_testing), labels_testing)
	params = ['chosen grid search cross validation params:', gridsearch_cv_params] if gridsearch_cv_params is not None else []
	statistic_training = ['training distribution:', dataset_training[dimension].value_counts()]
	statistic_testing = ['testing distribution:', dataset_test[dimension].value_counts()]
	
	logs = results + params + statistic_training + statistic_testing
	write_log_file(dataset.name, dimension, approach.name, logs)


if __name__ == '__main__':
	# reset_seed()
	
	frac_training = 0.8
	frac_validation = 0.0
	frac_test = 0.2

	# change image paths of all entries that are class==0 for the first dimension (readability) to a black dummy image;
	# the goal is that approx half of the images are black to increase contrast for testing if the training works
	is_run_as_dummy = False

	# download nltk once
	nltk.download('stopwords', quiet=True)

	# dataset setups
	dataset_path = os.path.join('dataset')
	datasets = [Dataset('open source dataset', dataset_path, ['labels.csv'], is_run_as_dummy)]

	# iterate over dataset setups (open source parts, all parts)
	for dataset in datasets:
		print_section('dataset: ' + dataset.name)
		dataset.shuffle()
		dataset.print_statistics()

		# iterate over dataset interpretations (Likert scale and binary)
		for dimensions, classes in [
			(dataset.dimensions_highest_probability_class, dataset.classes), 
			(dataset.dimensions_highest_probability_class_simplified, dataset.classes_simplified)]:

			# iterate over dimensions
			for dimension in dimensions:
				print_section('dimension: ' + dimension)

				run_approaches_bert(dataset, dimension, BERT(len(classes)), frac_training, frac_test, frac_validation)
				run_approaches_bert(dataset, dimension, RoBERTa(len(classes)), frac_training, frac_test, frac_validation)
				run_approaches_bert(dataset, dimension, CodeBERT(len(classes)), frac_training, frac_test, frac_validation)
				run_approaches_other(dataset, dimension, NaiveBayesTextBased(), 'code', frac_training, frac_test)
				run_approaches_other(dataset, dimension, SVMTextBased(), 'code', frac_training, frac_test)
				run_approaches_other(dataset, dimension, SVMImageBased(), 'image_path', frac_training, frac_test)
				run_approaches_other(dataset, dimension, AlexNet(len(classes)), 'image_path', frac_training, frac_test)

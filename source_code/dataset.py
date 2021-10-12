import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import reduce
from utils import split_into_chunks
from sklearn.model_selection import train_test_split


class Dataset:
	def __init__(self, name, path, labels_files_names, is_dummy=False):
		self.name = name
		self.path = path
		label_files_paths = self.get_labels_files_paths(path, labels_files_names)
		
		self.schema = [
			'projectname', 'packageandclass', 'path', 
			'readability', 'understandability', 'complexity', 
			'modularization', 'overall'
		]

		self.dimensions = ['readability', 'understandability', 'complexity']
		self.dimensions_original = list(map(lambda d: d + '_original', self.dimensions))
		
		self.dimensions_highest_probability_class = \
		list(map(lambda d: d + ' (highest probability class)', self.dimensions))
		
		self.dimensions_highest_probability_class_simplified = \
		list(map(lambda d: d + ' (highest probability class simplified)', self.dimensions))
		
		self.dimensions_classes_expectation_value = \
		list(map(lambda d: d + ' (class expectation value)', self.dimensions))
		
		# class indices from 0 to 3:
		self.classes = ['strongly agree', 'weakly agree', 'weakly disagree', 'strongly disagree']
		# class indices implified (binary):
		self.classes_simplified = ['agree', 'disagree']

		# read labels files
		column_indices = [self.schema.index(col) for col in ['path', *self.dimensions]]
		self.dataframe = self.read_labels_file(label_files_paths.pop(0), column_indices)
		while len(label_files_paths) > 0:
			further_dataframe = self.read_labels_file(label_files_paths.pop(0), column_indices)
			self.dataframe = self.dataframe.append(further_dataframe, ignore_index=True)

		# change dimension column names to dimensionname_original
		self.dataframe = self.dataframe.rename(columns=dict(zip(self.dimensions, self.dimensions_original)))

		# read code from file path as string
		self.dataframe['code'] = self.dataframe['path'].apply(self.read_code_file)

		self.set_class_interpretations()
		self.set_images_paths()

		if is_dummy:
			self.change_to_dummy()


	def get_labels_files_paths(self, path, labels_files_names):
		return [os.path.join(path, label_file_name) for label_file_name in labels_files_names]


	def read_labels_file(self, file_path, chosen_column_indices):
		return pd.read_csv(file_path, sep=';', usecols=chosen_column_indices)


	def read_code_file(self, file_path):
		file_path = os.path.join(self.path, *file_path.split('\\'))
		file_content = open(file_path, 'r', encoding='ISO-8859-1').read()
		return file_content


	def get_images_paths(self, file_path_source, image_folder_name):
		#  finds image_path that belongs to the code at file_path
		file_path = os.path.join(self.path, *file_path_source.split('\\'))
		image_path = file_path.replace('dataset', image_folder_name) + '.pdf.png'
		if os.path.isfile(image_path): 
			return image_path
		else:
			raise Exception('Could not find ' + image_path)


	def set_class_interpretations(self):
		for i in range(len(self.dimensions)):
			# Take the argmax of the Likert scale ratings, thus resulting in 4 possible discrete classes 
			# (strongly agree, weakly agree, weakly disagree, strongly disagree) => the dataset is very imbalanced
			self.dataframe[self.dimensions_highest_probability_class[i]] = \
			self.dataframe[self.dimensions_original[i]].apply(self.get_index_highest_prob_class)
			
			# Binary approach to cope with imbalance: the 3 least represented classes are merged into 1 class, 
			# such that the result are 2 discrete classes (agree, disagree) that are roughly balanced
			self.dataframe[self.dimensions_highest_probability_class_simplified[i]] = \
			self.dataframe[self.dimensions_highest_probability_class[i]].apply(
				lambda c: self.get_index_highest_prob_class_simplified(c, self.dimensions[i]))
			
			# Calculate the expectation value of the given probabilites for the Likert scale ratings
			self.dataframe[self.dimensions_classes_expectation_value[i]] = \
			self.dataframe[self.dimensions_original[i]].apply(self.get_class_expectation_value)


	def set_images_paths(self):
		self.dataframe['image_path'] = self.dataframe['path'].apply(lambda p: \
			self.get_images_paths(p, 'dataset_images'))
		# self.dataframe['image_normalized_path'] = self.dataframe['path'].apply(lambda p: \
		# 	self.get_images_paths(p, 'dataset_images_normalized'))


	def get_index_highest_prob_class(self, class_probabilities_string):
		# calculate the index of the class with the highest probability
		probabilities = self.parse_class_probabilities(class_probabilities_string)
		index = np.argmax(probabilities)
		return index


	def get_index_highest_prob_class_simplified(self, class_index, dimension):
		# similar to get_index_highest_prob_class; this time merging the 3 least popular classes into 1 class;
		# for complexity, the first 3 classes are the minority, for the other dimensions the last 3 classes
		agree_indices = [0, 1, 2] if dimension == 'complexity' else [0]
		# based on self.classes_simplified
		binary_class = 0 if class_index in agree_indices else 1
		return binary_class


	def get_class_expectation_value(self, class_probabilities_string):
		# calculate the expectation value of the classes
		indices = range(len(self.classes))
		probabilities = self.parse_class_probabilities(class_probabilities_string)
		expected_value = reduce(lambda acc, t: acc + t[0] * t[1], zip(indices, probabilities), 0)
		return expected_value


	def parse_class_probabilities(self, class_probabilities_string):
		# class_probabilities_string looks like this: {4.38897e-007,0.848266,0.151734,3.13027e-008}
		number_strings = class_probabilities_string.replace('{', '').replace('}', '').split(',')
		probabilities = list(map(float, number_strings))
		return probabilities


	def shuffle(self, SEED=42):
		self.dataframe = self.dataframe.sample(frac=1, random_state=SEED).reset_index(drop=True)


	def stratify_split(self, frac_training, frac_validation, frac_test, stratify_col_name):
		""" adapted from https://stackoverflow.com/a/65571687 """

		x = self.dataframe # Contains all columns
		y = self.dataframe[[stratify_col_name]] # Dataframe of just the column on which to stratify

		# Split original dataframe into train and temp dataframes
		df_training, df_temp, y_train, y_temp = \
		train_test_split(x, y, stratify = y, test_size = (1.0 - frac_training), random_state=None)

		if frac_validation == 0.0:
			return df_training, df_temp
		else:
			# Split the temp dataframe into val and test dataframes
			relative_frac_test = frac_test / (frac_validation + frac_test)
			df_validation, df_test, y_val, y_test = \
			train_test_split(df_temp, y_temp, stratify=y_temp, test_size=relative_frac_test, random_state=None)

			return df_training, df_validation, df_test


	def change_to_dummy(self):
		# change image paths of all entries that are class==0 for the first dimension (readability) to a black dummy image;
		# the goal is that approx half of the images are black to increase contrast for testing if the training works
		condition = self.dataframe[self.dimensions_highest_probability_class[0]] == 0
		self.dataframe.loc[condition, 'image_path'] = os.path.join('dataset_images', 'black_dummy.png')


	def bert_tokenize(self, tokenizer, target_column_name):
		# if already tokenized, don't do it again
		if target_column_name in self.dataframe.columns:
			return

		print('bert tokenization...')

		# work in another dataframe as a workaround to use object data type
		encoding_temp = pd.DataFrame(columns=[target_column_name])
		encoding_temp[target_column_name] = encoding_temp[target_column_name].astype(object)
		# input is limited to 512 tokens, therefore we split up each code into chunks 
		# and keep track of all chunks including the other information in a new dataframe
		dataframe_extended = pd.DataFrame().reindex_like(self.dataframe)
		dataframe_extended_current_length = 0
		for index, row in self.dataframe.iterrows():
			tokens = tokenizer.tokenize(row['code'])
			tokens_chunks = split_into_chunks(tokens, 506) # leave space for special chars
			for tokens in tokens_chunks:
				correctly_sized_string = tokenizer.convert_tokens_to_string(tokens)
				encoding = tokenizer.encode_plus(correctly_sized_string, padding='max_length')
				dataframe_extended.loc[dataframe_extended_current_length] = row
				encoding_temp.loc[dataframe_extended_current_length, target_column_name] = np.array(encoding.input_ids)
				dataframe_extended_current_length += 1
		dataframe_extended[target_column_name] = encoding_temp[target_column_name]

		# somehow it appends ".0" to the old floats, which we need to revert:
		for dimension in self.dimensions_highest_probability_class + self.dimensions_highest_probability_class_simplified:
			dataframe_extended[dimension] = dataframe_extended[dimension].astype(int)

		self.dataframe = dataframe_extended

		print(self.dataframe)


	def print_statistics(self, show_scatter_plot=False):
		# dataframe
		print(self.dataframe)

		# number of times each class occurs per dimension
		for dimension in self.dimensions_highest_probability_class + self.dimensions_highest_probability_class_simplified:
			print(self.dataframe[dimension].value_counts(), '\n')

		# scatter plot for regression variant
		if show_scatter_plot:
			mpl.rcParams['figure.dpi'] = 300
			plt.figure(figsize=(4, 10))
			plt.xticks(range(len(self.dimensions)), self.dimensions)
			plt.yticks(range(len(self.classes)), self.classes)
			plt.gca().invert_yaxis()
			for index, dimension in enumerate(self.dimensions_classes_expectation_value):
				values = self.dataframe[dimension]
				plt.plot(np.zeros_like(values) + index, values, '_')
			plt.show()

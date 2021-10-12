import os
import cv2
import time
import numpy as np
import pandas as pd
import random
import torch
from skimage.io import imread


def flatten_list(l):
	""" source: https://coderwall.com/p/rcmaea/flatten-a-list-of-lists-in-one-line-in-python """
	return [y for x in l for y in x]


def read_image(file_path):
	""" read image and convert from rgba to rbg """
	return np.delete(imread(file_path), 3, axis=2)


def resize_image(file_path, length):
	""" resizes squared image to length, returns image """
	return cv2.resize(cv2.imread(file_path), dsize=(length, length))


def split_camel_case(word):
	""" source: https://stackoverflow.com/a/37697078 """
	return re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word)).split()


def split_into_chunks(l, n):
	""" split list into chunks with each of it being < n long. 
	source: https://www.geeksforgeeks.org/break-list-chunks-size-n-python """
	return [l[i * n:(i + 1) * n] for i in range((len(l) + n - 1) // n )]


def print_section(string):
	print('\n\n\n==========================\n', string, '\n==========================\n\n\n')


def write_log_file(dataset_name, dimension_name, approach_name , strings):
	""" write text given as array of strings into log file with 
	the name file_name_info prepended by the unix timestamp """
	filename = str(int(time.time())) + '_' + dataset_name + '_' + dimension_name + '_' + approach_name + '.txt'
	if not os.path.exists('logs'):
		os.makedirs('logs')
	file_path = os.path.join('logs', filename)
	np.savetxt(file_path, strings, fmt='%s')
	print('wrote log file:', file_path)


def reset_seed(SEED=42):
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)

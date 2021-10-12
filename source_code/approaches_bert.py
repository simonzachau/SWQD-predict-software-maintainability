import pandas as pd
import tensorflow as tf
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from utils import print_section


class BERTApproach:
	def __init__(self, name, model, tokenizer, padding_token_id):
		self.name = name
		self.padding_token_id = padding_token_id

		self.BATCH_SIZE = 8
		self.NUM_EPOCHS = 4

		# GPU/CPU
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.tokenizer = tokenizer()
		self.model = model()


	# the following is adapted from 
	# https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1
	# and https://mccormickml.com/2019/07/22/BERT-fine-tuning/#31-bert-tokenizer
	# and https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b


	def create_attention_masks(self, input_ids):
		# if the input id is the padding token id, use 0, else use 1
		return list(map(lambda segment: [int(token_id != self.padding_token_id) for token_id in segment], input_ids))


	def create_dataloader(self, input_ids, labels=[]):
		inputs = [input_ids, self.create_attention_masks(input_ids)]
		if len(labels) > 0:
			inputs.append(labels)
		data = TensorDataset(*map(torch.tensor, inputs))
		dataloader = DataLoader(data, sampler=RandomSampler(data), batch_size=self.BATCH_SIZE)
		return dataloader


	def train(self, input_ids_training, labels_training, input_ids_validation=[], labels_validation=[]):
		print_section('train model: ' + self.name)

		dataloader_training = self.create_dataloader(input_ids_training, labels=labels_training)
		dataloader_validation = self.create_dataloader(input_ids_validation, labels=labels_validation) if len(labels_validation) != 0 else None

		eval_every = len(dataloader_training) // 2
		best_valid_loss = float('Inf')

		self.model.to(self.device)
		optimizer = optim.Adam(self.model.parameters(), lr=2e-5)
	
		# initialize running values
		running_loss = 0.0
		valid_running_loss = 0.0
		global_step = 0
		train_loss_list = []
		valid_loss_list = []
		global_steps_list = []

		# training loop
		self.model.train()
		for epoch in range(self.NUM_EPOCHS):
			for step, batch in enumerate(dataloader_training):
				inputs = {'input_ids':		batch[0].to(self.device),
						  'attention_mask': batch[1].to(self.device),
						  'labels':			batch[2].to(self.device)}

				outputs = self.model(**inputs)
				loss = outputs[0]

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# update running values
				running_loss += loss.item()
				global_step += 1

				# evaluation step
				if global_step % eval_every == 0:
					self.model.eval()
					with torch.no_grad():

						# validation loop
						if len(labels_validation) != 0:
							for batch in dataloader_validation:
								inputs = {'input_ids':		batch[0].to(self.device),
										  'attention_mask': batch[1].to(self.device),
										  'labels':			batch[2].to(self.device)}

								outputs = self.model(**inputs)
								loss = outputs[0]
								
								valid_running_loss += loss.item()

					# evaluation
					average_train_loss = running_loss / eval_every
					average_valid_loss = valid_running_loss / len(dataloader_validation) if len(labels_validation) != 0 else 0
					train_loss_list.append(average_train_loss)
					valid_loss_list.append(average_valid_loss)
					global_steps_list.append(global_step)

					# resetting running values
					running_loss = 0.0
					valid_running_loss = 0.0
					self.model.train()

					if len(labels_validation) != 0:
						print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
							  .format(epoch + 1, self.NUM_EPOCHS, global_step, self.NUM_EPOCHS * len(dataloader_training),
									  average_train_loss, average_valid_loss))
					else:
						print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'
							  .format(epoch + 1, self.NUM_EPOCHS, global_step, self.NUM_EPOCHS * len(dataloader_training),
									  average_train_loss))
		print('Finished Training!')


	def test(self, input_ids_testing):
		print_section('test model: ' + self.name)

		predictions = []

		dataloader_testing = self.create_dataloader(input_ids_testing)

		self.model.eval()
		with torch.no_grad():
			for batch in dataloader_testing:
				inputs = {'input_ids':		batch[0].to(self.device),
						  'attention_mask': batch[1].to(self.device)}

				outputs = self.model(**inputs)
				output = outputs[0]

				predictions.extend(torch.argmax(output, 1).tolist())

		return predictions


class BERT(BERTApproach):
	def __init__(self, num_classes):
		model = lambda: BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
		tokenizer = lambda: BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
		super().__init__('BERT', model, tokenizer, 0)


class RoBERTa(BERTApproach):
	def __init__(self, num_classes):
		model = lambda: RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_classes)
		tokenizer = lambda: RobertaTokenizer.from_pretrained('roberta-base')
		super().__init__('RoBERTa', model, tokenizer, 1)


class CodeBERT(BERTApproach):
	def __init__(self, num_classes):
		model = lambda: RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=num_classes)
		tokenizer = lambda: RobertaTokenizer.from_pretrained('microsoft/codebert-base')
		super().__init__('CodeBERT', model, tokenizer, 1)

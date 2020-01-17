import pandas as pd
import os
import pickle as pkl
from random import randint
from tqdm import tqdm_gui

def load():
	print("loading the dataset")
	col_names = ["Query", "Label", "In/Out"]
	df = pd.read_excel('./FoodBot Intents (1).xlsx', names=col_names)
	queries = list(df['Query'])
	labels = list(df['In/Out'])
	l1 = list(zip(queries, labels))
	df = pd.read_excel('./Quinto Intents (1).xlsx', names=col_names)
	queries = list(df['Query'])
	labels = list(df['In/Out'])
	return l1, list(zip(queries, labels))


def rand_name(name_len=10):
	choices = list(
	    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
	return ''.join([choices[randint(0, len(choices)-1)] for _ in range(name_len)])


def gen_sample(botID, dataset_name=None, train_set=None, eval_set=None):
	data_dir = './data'
	if not os.path.exists(data_dir):
		os.mkdir(data_dir)
	# if not train_set and not eval_set:
	# 	train_set, eval_set = load()
	model_data_dir = os.path.join(data_dir, botID)
	if not os.path.exists(model_data_dir):
		os.mkdir(model_data_dir)
	random_sample_name = dataset_name or rand_name()
	try:
		if dataset_name:
			dataset_path = os.path.join(model_data_dir, random_sample_name)
			print("dataset path :", dataset_path)
			with open(dataset_path, 'rb') as f:
				dataset = pkl.load(f)
			print("loaded dataset.")
			train_set, eval_set = dataset["train_dataset"], dataset["eval_dataset"]
	except Exception as e:
		print(e)
	else:
		print("Current dataset name : ", random_sample_name)
		print("Dumping the dataset")
		with open(os.path.join(model_data_dir, random_sample_name), 'wb') as f:
			pkl.dump({"train_dataset": train_set, "eval_dataset": eval_set}, f)
		with open(os.path.join(model_data_dir, 'data_history.txt'), 'a+') as f:
			f.write(random_sample_name + "\n")
	return random_sample_name, train_set, eval_set


if __name__ == "__main__":
	name, train_set, eval_set = gen_sample("platform_data", dataset_name="b4NweBkwTr")
	with open("train.pkl", "wb") as fi:
		pkl.dump(train_set, fi)
	with open("eval.pkl", "wb") as fi:
		pkl.dump(eval_set, fi)

# https://storage.googleapis.com/glib-sic/output.tar.gz
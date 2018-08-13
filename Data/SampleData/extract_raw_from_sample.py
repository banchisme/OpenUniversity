import pandas as pd 
import os

def extract_sample(f_path):
	f_name_short, f_type = f_name.split('.')

	# only csv is impleted
	if f_type == 'csv':
		df = pd.read_csv(f_name + '.' + f_type)

if __name__ == '__main__':
	# take the first 10 lines of code as sample data
	for f in os.listdir('../Raw'):
		if os.path.isfile(f):


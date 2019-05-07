import pandas as pd
import os
import shutil as st 
import argparse
import sys
def main(args):
	exel = pd.read_csv(args.csv_file)
	path = args.path
	if not os.path.exists(path):
		os.mkdir(path)
	array_label = exel['label'].values
	array_image = exel['image'].values
	#print(array[1])
	for i in range (len(array_label)):
		dirname = os.path.join(path, '[' +str(array_label[i]) + ']')
		if not os.path.exists(dirname):
			os.mkdir(dirname)


	for i in range(len(array_image)):
		copypath = os.path.join(args.train_data, array_image[i])
		tmp = exel[exel.image == str(array_image[i])]
		print(os.path.join(path, str(tmp['label'].values)))
		st.copy2(copypath, os.path.join(path, str(tmp['label'].values)))

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('path', type= str, help= 'Foler\'s Name of data to train')
	parser.add_argument('csv_file', type = str, help = 'csv file')
	parser.add_argument('train_data', type = str, help = 'train_data')
	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
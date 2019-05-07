import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image 
import os
import glob
import argparse
import sys

def mtcnn_image(image, margin, image_size):
	detector = MTCNN()
	imageObject = Image.open(image)
	image = mpimg.imread(image)
	result = detector.detect_faces(image)
	for person in result:
		bounding_box = person['box']
		keypoints = person['keypoints']
	cropped = imageObject.crop((bounding_box[0] - margin, bounding_box[1]  - margin ,bounding_box[0]+bounding_box[2] + margin, bounding_box[1] + bounding_box[3] + margin))
	cropped = cropped.resize((image_size, image_size), 0)

	return cropped

def save_aligned_image(input_dir, output_dir, margin, image_size):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	os.chdir(input_dir)
	for infile in glob.glob("*.jpg"):
	    file, ext = os.path.splitext(infile)
	    os.chdir(input_dir)
	    os.chdir(output_dir)
	    mtcnn_image(os.path.join(input_dir, infile), margin, image_size).save(file +".jpg")
	    print('Success aligned ' + infile)


def save_image(input_dir, output_dir, margin, image_size):
	os.chdir(input_dir)
	dir = os.listdir(input_dir)
	for file in dir:
		save_aligned_image(os.path.join(input_dir, file), os.path.join(output_dir, file), margin, image_size)

def main(args):
	#output_dir = os.path.exists(args.output_dir)
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	save_image(args.input_dir, args.output_dir, args.margin, args.size)


def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('input_dir', type = str, help ='Directory with unaligned image.')
	parser.add_argument('output_dir', type = str, help = 'Directory with aligned face image.')
	parser.add_argument('--margin', type = int, help = 'Margin for the crop around the bounding box.', default = 20)
	parser.add_argument('--size', type = int, help = 'Image size after aligned')

	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))


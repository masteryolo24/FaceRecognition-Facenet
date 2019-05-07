import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
import pandas as pd 
import csv
from pandas import DataFrame

def create_csvfile(csv_file_path, title):
    with open(csv_file_path, mode = 'w') as file:
        writer = csv.writer(file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)

        writer.writerow(title)

def save_csv(result, csv_file_path, title, images_paths, number_column, predictions):
    #result = pd.read_csv(csv_file_path)
    result[title[0]] = images_paths

    result[title[number_column]] = predictions
    result.to_csv(csv_file_path)
    return result

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            np.random.seed(seed=args.seed)
            dataset = facenet.get_dataset(args.data_dir)

            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            

            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
            
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                print('Loaded classifier model from file "%s"' % classifier_filename_exp)
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)

                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)
                title = ['image']
                for i in range(0, len(class_names)):
                    title.append(class_names[i])
                create_csvfile(args.csv_file_path, title)
                result = pd.read_csv(args.csv_file_path)

                for i in range(0, predictions.shape[1]):
                    array = []
                    for j in range(0, len(paths)):
                        array.append(predictions[j][i])
                    save_csv(result, args.csv_file_path, title, paths, i+1, array)                

                #save_csv(args.csv_file_path, title, paths)

                ##############################################
"""
                for i in range (0, len(paths)):
                    print(paths[i])
                    for j in range (0, predictions.shape[1]):
                        print(predictions[i][j])
                print(predictions[0][0])"""



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--csv_file_path', type=str, help = 'Patch of csv file.', default = 'facenet.csv')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
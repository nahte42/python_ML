import tensorflow as tf
import os
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe


#tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
#print("Eager execution: {}".format(tf.executing_eagerly()))

#First we need to save the dataset

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

"""
The dataset is composed of one example per line
Features within the first four fields
And lastly labels, which is what we want to predict
0 --- Iris Setosa
1 --- iris versicolor
2 --- Iris Virginica
"""

#Next we want to parse the dataset

def parse_csv(line):
    example_defaults = [[0.],[0.],[0.],[0.],[0.]] #This sets the field typres
    parsed_line = tf.defcode_csv(line, example_defaults)
    #Remember that the first 4 fields of the data are features
    features = tf.reshape(parsed_line[:-1], shape = (4,))
    #Last field is the label
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label

#The next part is to create the training dataset


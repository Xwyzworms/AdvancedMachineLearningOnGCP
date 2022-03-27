#%%
import os
import shutil
import importlib
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import  (Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Softmax)
from . import util
importlib.reload(util)

# CONST VARIABLE

WIDTH = 28
HEIGHT = 28


def get_layers( model_type, nclasses = 10, hidden_layer_1_neurons=400,
            	hidden_layer_2_neurons=100, dropout_rate = 0.25, num_filters1=64,
             	kernel_size1=3, pooling_size1= 2, num_filters2=32, kernel_size2=3,pooling_size2=2):
    model_layers = {
		"linear":[
				Flatten(),
				Dense(nclasses,activation='softmax')
    ],
		"dnn" : [
				Flatten(),
				Dense(hidden_layer_1_neurons, activation='relu'),
				Dense(hidden_layer_2_neurons, activation=tf.nn.relu),
				Dense(nclasses, activation='softmax')
    ],
		"dnn_dropout" :[
				Flatten(),
				Dense(hidden_layer_1_neurons, activation='relu'),
				Dropout(dropout_rate),
				Dense(hidden_layer_2_neurons, activation=tf.nn.relu),
				Dense(nclasses, activation = tf.nn.softmax)
    ],
		"cnn" : [
				Conv2D(filters=num_filters1, kernel_size=kernel_size1, activation="relu", input_shape=(WIDTH, HEIGHT, 1)),

				MaxPooling2D(pool_size=pooling_size1),

				Conv2D(filters=num_filters2, kernel_size=kernel_size2, activation="relu" ),

				MaxPooling2D(pool_size=pooling_size2),
				Flatten(),
				Dense(hidden_layer_1_neurons, activation='relu'),
				Dense(hidden_layer_2_neurons, activation="relu"),
				Dropout(dropout_rate),
				Dense(nclasses, activation='softmax')
		]
	}  
    
    return model_layers[model_type]


def build_model(layers, output_dir):
    model = Sequential(layers)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train_and_evaluate(model, num_epochs, steps_per_epoch, output_dir):
    mnist = tf.keras.datasets.mnist.load_data()
    train_data = util.load_dataset(mnist)
    validation_data = util.load_dataset(mnist, training=False)
    
    callbacks : List = []
    if output_dir:
     	tensorboard_callback = TensorBoard(log_dir=output_dir)
     	callbacks.append(tensorboard_callback)
     
    history = model.fit(train_data, epochs=num_epochs, steps_per_epoch=steps_per_epoch, validation_data=validation_data,
    callbacks = callbacks)
    if output_dir:
        export_path = os.path.join(output_dir, "keras_export")
        model.save(export_path, save_format="tf")
    return history


#if __name__ == "__main__":
#    model = build_model(get_layers("dnn"),"")
#    train_and_evaluate(model, 1, 100, "test")
	
# %%

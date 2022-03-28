#%%
import os
import subprocess

PROJECT : str = "qwiklabs-gcp-03-991127cfa651"
BUCKET : str = "qwiklabs-gcp-03-991127cfa651"
REGION : str = "us-central1"



os.environ["PROJECT"] = PROJECT
os.environ["BUCKET"] = BUCKET
os.environ["REGION"] = REGION
os.environ["TFVERSION"] = "2.6.2"


subprocess.Popen(["setGcloud.sh {} {}".format(PROJECT, REGION)], shell=True)

# %%

## Do something useful here

import pathlib
import IPython.display as display
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D)


# %%
data_directory = tf.keras.utils.get_file("flower_photos","https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz", untar=True)

print("cd", data_directory)
# %%
print(data_directory)
data_dir = pathlib.Path(data_directory)
print(type(data_dir)) # lokasi dari datanya

CLASS_NAMES : np.array = np.array([item.name for item in data_dir.glob("*") if item.name != "LICENSE.txt"])

print("Here are the class names: {}".format(CLASS_NAMES))


# %%
# if mau display image yang didapet
roses = list(data_dir.glob("roses/*")) ## Ini ngerubahnya jadi lebih enak aja

for image_path in roses[:5]:
    display.display(Image.open(str(image_path)))

# %%

subprocess.run("bash checkData.sh",shell=True,stdout=subprocess.PIPE).stdout
# %%
IMG_HEIGHT : int = 224
IMG_WIDTH : int = 224
IMG_CHANNELS : int = 3

BATCH_SIZE = 32

SHUFFLE_BUFFER = 10* BATCH_SIZE 

AUTOTUNE = tf.data.experimental.AUTOTUNE

VALIDATION_IMAGES = 370
VALIDATION_STEPS = VALIDATION_IMAGES // BATCH_SIZE

def decode_img(img, reshape_dims):
    # Convert String to uint8 Tensor
    img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)
    ## Convert to floats in [0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    ## Resize
    return tf.image.resize(img, reshape_dims)

img = tf.io.read_file("gs://cloud-ml-data/img/flower_photos/daisy/754296579_30a9ae018c_n.jpg")


def decode_csv(csv_row):
    record_defaults = ["path", "flower"]
    filename, label_string = tf.io.decode_csv(csv_row, record_defaults=record_defaults)
    image_bytes = tf.io.read_file(filename)
    label = tf.math.equal(CLASS_NAMES , label_string)
    return image_bytes, label


#print(img)
img = decode_img(img, [IMG_WIDTH, IMG_HEIGHT])
plt.imshow((img.numpy()))



MAX_DELTA = 0.1 # Less Brightness if the contrast is 100, then i reduced it by half
CONTRAST_LOWER = 0.2  # Lower ound
CONTRAST_UPPER = 1.8 # Upper boiuynd ( COntrastnya lebih contrast lagi karena diatas 1)

def read_and_preprocess(image_bytes, label, random_augment=False):
    if random_augment:
        img = decode_img(image_bytes, [IMG_HEIGHT + 10, IMG_WIDTH + 10]) # Diperbesar imagenya ( Biar nanti pas augment data yang penting ga hilang)
        img = tf.image.random_crop(value=img, size= [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS] )
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, MAX_DELTA)
        img = tf.image.random_contrast(img, CONTRAST_LOWER, CONTRAST_UPPER)
    else :
        img = decode_img(image_bytes, [IMG_WIDTH, IMG_HEIGHT])
        
    return img, label


def read_and_preprocess_with_augment(image_bytes, label):
    return read_and_preprocess(image_bytes, label)

## Reading and processing on Memory , not storing it

def load_dataset(csv_of_filenames, batch_size, training=True):
    dataset = tf.data.TextLineDataset(csv_of_filenames).map(decode_csv).cache()
    
    if training:
        dataset = dataset \
        .map(read_and_preprocess_with_augment) \
        .shuffle(SHUFFLE_BUFFER) \
        .repeat(count=None) # Repeat forever, All Photos Will be repeated during Trianing ( Setiap Batch Mungkin akan dapat data yang sama kek sebelumnya )
    else:
        dataset = dataset.map(read_and_preprocess) \
        .repeat(count=1) # Each photo will only be used once during evaluation
    
    # Prefetch data to improve speed ( Siapkan batch selanjutnya, ketika sedang raining)
    return dataset.batch(batch_size=batch_size).prefetch(buffer_size=AUTOTUNE)
# %%
## Run it

train_data = "gs://cloud-ml-data/img/flower_photos/train_set.csv"
train_data = load_dataset(train_data,1) # Per batch only 1 Observation
iterate = iter(train_data)

image_batch, label_batch = next(iterate)
img = image_batch[0]
plt.imshow(img)
print(label_batch[0])


# %%

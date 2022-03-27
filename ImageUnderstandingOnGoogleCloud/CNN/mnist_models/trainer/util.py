import tensorflow as tf

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255 # range 1 - 0
    image = tf.expand_dims(image, 1) # add a new dimension ( x,x, 1)
    return image, label



def load_dataset(
	data, training : bool = True, buffer_size : int = 5000, batch_size : int = 100, nclasses : int = 10 ):
    
    (x_train, y_train), (x_test, y_test) = data
    x  = x_train if training else x_test
    y = y_train if training else y_test
    
    y = tf.keras.utils.to_categorical(y, nclasses)
    dataset =tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.map(scale).batch(batch_size)
    if training : 
        dataset = dataset.shuffle(buffer_size).repeat()
    return dataset
    
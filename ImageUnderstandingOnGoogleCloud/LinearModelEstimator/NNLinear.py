#%%
import importlib
import shutil
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)



#%%

# EXPLORING DATA
model = None
HEIGHT : int = 28
WIDTH : int = 28
NCLASSES : int = 10

## Preparing data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255 , x_test / 255

y_train = tf.keras.utils.to_categorical(y_train, NCLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NCLASSES)

print("x_train.shape = {}".format(x_train.shape))
print("y_train.shape = {}".format(y_train.shape))
print("x_test.shape = {}".format(x_test.shape))
print("y_test.shape = {}".format(y_test.shape))



# %%
def linear_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(HEIGHT, WIDTH, 1),name="image"))
    model.add(tf.keras.layers.Flatten(name="flatten"))
    model.add(tf.keras.layers.Dense(units=NCLASSES, activation=tf.nn.softmax, name="outputProbabilities"))
    return model

train_input_obj = tf.estimator.inputs.numpy_input_fn(
	x = {"image" : x_train},
	y = y_train,
	batch_size = 100,
	num_epochs = None, # None means infinite ( Nanti dipake di Distrubsi ML)
	shuffle = True, # Shuffle the data ( Karena ingin dilakukan evaluasi), jadi ada baiknya di kocok
	queue_capacity = 5000
)

eval_input_obj = tf.estimator.inputs.numpy_input_fn(
	x = {"image" : x_test},
	y = y_test,
	batch_size = 100,
	num_epochs=1, # Cuman Ingin 1 Kali aja untuk melakukan Evaulias
	shuffle= False,
	queue_capacity = 5000
)

# Serving input ini digunakan untuk melakukan prediksi ( Hal pertama adalah buat placeholdernya dahulu untuk menyimpann prediksi sebelum dikalkulasikan dengan loss)
def serving_input_fn() :
    placeholders = {"image" : tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, 1])}
    features = placeholders
    return tf.estimator.export.ServingInputReceiver(features = features, receiver_tensors = placeholders)

def train_and_evaluate(output_dir , hparams):
    
    model = linear_model()
    
    model.compile(
		optimizer = "adam",
		loss = "categorical_crossentropy",
		metrics =["accuracy"])
	# Convert Keras Model ke ESTIMATOR
    estimator = tf.keras.estimator.model_to_estimator(keras_model = model, model_dir=output_dir)
    
    train_spec = tf.estimator.TrainSpec(
		input_fn = train_input_obj,
		max_steps = hparams["train_steps"])
    
    # Exporter digunakan untuk Melakukan prediksi model dengan menggunakan Serving input fn
    exporter = tf.estimator.LatestExporter("exporter", serving_input_fn)
    
    eval_spec = tf.estimator.EvalSpec(
		input_fn = eval_input_obj,
		steps=None,
		exporters = exporter)
    
    tf.estimator.train_and_evaluate(
        estimator = estimator, 
        train_spec = train_spec, 
        eval_spec = eval_spec)
    
    print(model.predict(x_test[5].reshape(1, HEIGHT, WIDTH, 1)))
    
OUTDIR = "mnist\learned"
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

hparams = {"train_steps": 1000, "learning_rate": 0.01}
train_and_evaluate(OUTDIR, hparams)
# %%



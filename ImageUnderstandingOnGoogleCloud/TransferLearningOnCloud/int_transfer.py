#%%

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
# %%

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,test_labells) = fashion_mnist.load_data()

# %%
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape) # (60000, 28, 28)
print(len(train_labels)) # 60000
print(train_labels) # Not One Hot Encoded yet


#%%

plt.figure()
plt.imshow(train_images[2],cmap="gray")
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0


plt.figure(figsize=(15,15))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]],color="white") # Change to black if you using White Background
plt.show()

#%%
def getLayers():
    return [
		tf.keras.layers.Flatten(input_shape=(28, 28)),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(10) # Going to be used as input to Softmax( Logits )
	]

def compile(model):
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
    
model = tf.keras.Sequential(getLayers())
compile(model)
print(model.summary())

# %%

history = model.fit(train_images, train_labels, epochs=10)
# %%
test_loss,test_acc = model.evaluate(test_images, test_labells, verbose=2)
print("Test Acc {}".format(test_acc))

# %%
#dikarenakan modelnya itu sendiri akhirnya menggunakan si Linear output, so perlu dilakukan SOFTMAX() agar dapat dikonversi menjadi Probabilitas

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

# %%
predicts = probability_model.predict(test_images)
print(predicts[0])
# %%
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array,axis=0)
    
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"
        
    plt.xlabel("{} {:2.0f}% ({}) ".format(class_names[predicted_label],
                                           100*np.max(predictions_array),
                                           class_names[true_label], color=color))

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")
    
# %%
i=100
plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
plot_image(i, predicts[i], test_labells, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predicts[i], test_labells)
plt.show()
# %%
num_rows=6
num_cols=3
num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1) # Makan 2 tempat, ambil kolom yang 1
    plot_image(i, predicts[i], test_labells, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2) # Makan 2 tempat, ambil yang kolom 2nya
    plot_value_array(i, predicts[i], test_labells)
plt.show()
# %%

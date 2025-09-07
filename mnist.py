import tensorflow  as tf
import tensorflow_datasets as tsfd
#Lets disable the progress bar for cleaner output
tsfd.disable_progress_bar()

import math
import numpy as np
import matplotlib.pyplot as plt
import logging
#Getting tensorflow's default logger instance
logger = tf.get_logger()
#Set the logging level to only show ERROR
#This hides INFO and Warning logs from Tensorflow to keep output clean
logger.setLevel(logging.ERROR)
#Load the FASHION MNIST dataset from Tensorflow Datasets
# as_supervised=True → returns data as (image, label) pairs instead of a dict
# with_info=True → also returns metadata (like label names, number of examples, etc.)
dataset, metadata = tsfd.load('fashion_mnist', as_supervised=True, with_info=True)
#splitting dataset into training and testing sets
train_dataset, test_dataset = dataset['train'], dataset['test']
# Extract the human-readable class names from the dataset metadata
# metadata.features['label'].names → list of label names in order (0–9)
class_names = metadata.features['label'].names
print("Class names:{}".format(class_names))

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples: {}".format(num_test_examples))

#Function to normalize images: convert uint8 pixel values (0–255) to float32 (0–1)
def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)
# Cache the training dataset in memory after the first load
# This avoids re-reading/recomputing it from disk each epoch → speeds up training
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

for image, label in test_dataset.take(1):
  break
# Convert the TensorFlow tensor to a NumPy array and reshape it to 28x28 pixels
image = image.numpy().reshape((28,28))

plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))
i = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1
plt.show()

model = tf.keras.Sequential([
   # 1️⃣ Convolutional layer: 32 filters, each 3x3, same padding
    # - Extracts low-level features (edges, textures) from the 28x28 grayscale image
    # - 'same' padding keeps output size the same as input
    # - ReLU activation introduces non-linearity
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(28, 28, 1)),
    # 2️⃣ MaxPooling layer: reduces spatial dimensions by taking max in 2x2 windows
    # - Downsamples feature maps → reduces computation and helps generalization
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    # 3️⃣ Second convolutional layer: 64 filters, deeper feature extraction
    # - Learns more complex patterns (shapes, parts of clothing)
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    # 4️⃣ Second pooling layer: further reduces spatial size
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    # 5️⃣ Flatten: converts 2D feature maps into a 1D vector
    # - Prepares data for fully connected layers
    tf.keras.layers.Flatten(),
    # 6️⃣ Dense (fully connected) layer: 128 neurons
    # - Learns high-level combinations of features
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    # 7️⃣ Output layer: 10 neurons (one per class)
    # - Softmax activation outputs probability distribution over classes
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

test_loss, test_accuracy = model.evaluate(test_dataset, steps = math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)

for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)
predictions.shape
predictions[0]

np.argmax(predictions[0])

test_labels[0]

def plot_image(i, predictions_array, true_labels, images):
  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img[...,0], cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

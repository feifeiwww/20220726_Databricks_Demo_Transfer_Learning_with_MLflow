# Databricks notebook source
# MAGIC %md
# MAGIC This code is modified from the TensorFlow tutorial: https://www.tensorflow.org/tutorials/images/transfer_learning

# COMMAND ----------

# MAGIC %md
# MAGIC # Import Libraries

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow Autologging

# COMMAND ----------

mlflow.tensorflow.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Train, Validation, and Test Sets

# COMMAND ----------

# Set random seed
tf.keras.utils.set_random_seed(42)

# Define paths
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# Load train and validation sets
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# COMMAND ----------

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE
                                                            )

# COMMAND ----------

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE
                                                                 )

# COMMAND ----------

# Set a portion of the validation set aside for the test set
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
# Print dataset sizes
print(f"Train Dataset: {len(train_dataset)} batches")
print(f"Validation Dataset: {len(validation_dataset)} batches")
print(f"Test Dataset: {len(test_dataset)} batches")

# COMMAND ----------

# MAGIC %md
# MAGIC # Display Example Images

# COMMAND ----------

# Extract class names
class_names = train_dataset.class_names

# Display 9 example images with their class names
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# COMMAND ----------

# MAGIC %md
# MAGIC # Optimize Image Loading

# COMMAND ----------

# Use buffered prefetching to load images from disk - https://www.tensorflow.org/guide/data_performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# COMMAND ----------

# MAGIC %md
# MAGIC # Augment Data

# COMMAND ----------

# Increase generalizability of model adding flipped and rotates images to the dataset
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

# COMMAND ----------

# Display example of augmented images from an image in dataset
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Base Imagenet Model: MobileNetV2

# COMMAND ----------

# Load base imagenet model MobileNetV2 setting include_top=False so we drop the top layers
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# Freeze base weights so they don't retrain
base_model.trainable = False

# Display the base model architecture
base_model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC # Define Model

# COMMAND ----------

# Define preprocessing layers
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1) # Alternatively

# Define new prediction layers for cats vs. dogs classification
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

# Define model layers
inputs = tf.keras.Input(shape=(160, 160, 3)) # image input shape
x = data_augmentation(inputs) # apply image preprocessing layers
x = preprocess_input(x)
x = base_model(x, training=False) # apply base model, without retraining its weights
x = global_average_layer(x) # apply final prediction layers
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x) 

# Create model from model layers
model = tf.keras.Model(inputs, outputs)

# Compile model 
learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Display model summary
model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC # Train Model

# COMMAND ----------

# Train model for num_epochs
num_epochs = 5
history = model.fit(train_dataset,
                    epochs=num_epochs,
                    validation_data=validation_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC # Display Loss and Accuracy Graph

# COMMAND ----------

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Display Test Set Predictions

# COMMAND ----------

# Test set performance
loss, accuracy = model.evaluate(test_dataset)
print(f"Test set loss: {loss}, Test set accuracy: {accuracy}")

# COMMAND ----------

# Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")

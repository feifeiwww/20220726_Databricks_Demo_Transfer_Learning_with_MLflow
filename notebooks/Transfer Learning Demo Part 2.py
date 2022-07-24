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
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

# Set a portion of the validation set aside for the test set
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

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

# MAGIC %md
# MAGIC # Grid Search with MLflow Manual Logging

# COMMAND ----------

# Define function to plot model loss for MLflow run
def view_model_loss(history):
    plt.clf()
    plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    return plt

# COMMAND ----------

# Define MLflow parent run 
with mlflow.start_run(run_name="Parent Run") as parent_run:
    
    # Define hyperparameter grids
    learning_rates = [0.001, 0.0001]
    beta_1s = [0.5, 0.9]
    
    # Loop over learning rate values
    for lr in learning_rates:
        
        # Loop over beta_1 values
        for beta_1 in beta_1s:
            
            # Start MLflow run
            with mlflow.start_run(nested=True) as run:
                
                # Define model
                IMG_SHAPE = IMG_SIZE + (3,)
                base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
                base_model.trainable = False
                global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
                prediction_layer = tf.keras.layers.Dense(1)
                preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
                rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
                inputs = tf.keras.Input(shape=(160, 160, 3))
                x = data_augmentation(inputs)
                x = preprocess_input(x)
                x = base_model(x, training=False)
                x = global_average_layer(x)
                x = tf.keras.layers.Dropout(0.2)(x)
                outputs = prediction_layer(x)
                model = tf.keras.Model(inputs, outputs)
                
                # Compile model with learning rate and beta_1 value
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
                
                # Train model
                history = model.fit(train_dataset,
                            epochs=2,
                            validation_data=validation_dataset)

                
                
                # Create loss plot
                plt = view_model_loss(history)
                fig = plt.gcf()
                
                # Manually log to MLflow model, learning rate, beta_1, val_accuracy, and the loss curve plot
                mlflow.keras.log_model(model, "model")
                mlflow.log_params({"learning_rate": lr, "beta_1":beta_1})
                mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
                mlflow.log_figure(fig, "loss_curve.png")

# COMMAND ----------

# MAGIC %md
# MAGIC # Get Best Model Run

# COMMAND ----------

# Create MLflow tracking client
from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

# Display all runs
display(client.list_run_infos(run.info.experiment_id))

# COMMAND ----------

# Search for run with highest validation accuracy
best_run = client.search_runs(run.info.experiment_id, order_by=["metrics.val_accuracy desc"], max_results=1)
best_run

# COMMAND ----------

# Alternatively, get past runs as a Spark DataFrame and then query that instead
runs = spark.read.format("mlflow-experiment").load()
display(runs.orderBy("metrics.val_accuracy", ascending=False))

# COMMAND ----------

best_run_id = runs.orderBy("metrics.val_accuracy",ascending=False).first()["run_id"]
best_run_id

# COMMAND ----------

# MAGIC %md
# MAGIC # Register Best Model (Below are optional)
# MAGIC Note: model registry is currently not available on the free Databricks Community Edition, but is available on Databricks enterprise editions. If you would like to run code locally, you may need to config additional settings.

# COMMAND ----------

# # Register the best model to MLflow model registry
# model_uri = f"runs:/{best_run_id}/model"

# model_details = mlflow.register_model(model_uri=model_uri, name="demo_transfer_learning")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Model From Registry

# COMMAND ----------

# # Load in first version of model
# model_path = f"models:/demo_transfer_learning/1" 
# print(f"Loading registered model version from URI: '{model_path}'")

# # Load model
# loaded_model = mlflow.keras.load_model(model_path)

# COMMAND ----------

# # Display image to predict
# image_batch, label_batch = test_dataset.as_numpy_iterator().next()
# image_to_predict = image_batch[1]
# plt.imshow(image_to_predict.astype("uint8"))

# # Predict, values closer to 0 are cats, closer to 1 are dogs
# def cat_or_dog(im):
#     if tf.nn.sigmoid(model.predict(np.expand_dims(im, axis=0))).numpy()[0][0] < 0.5:
#         print("Cat!")
#     else:
#         print("Dog!")
# cat_or_dog(image_to_predict)

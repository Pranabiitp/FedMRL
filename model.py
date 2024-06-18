#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
# import tensorflow_federated as tff
import numpy as np

# Define a simple model
def create_model():
    base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        # Freeze all layers ""except the last two convolutional layers and the classification layer
#         for layer in base_model.layers[:-5]:
#             layer.trainable = False
    base_model.trainable=False
        # Create the transfer learning model by adding custom classification layers on top of the base model
    model2 = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
#             tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')  # Adjust the number of output classes accordingly
        ])

        # Compile the model
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model2


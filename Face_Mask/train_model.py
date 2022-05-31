import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, AveragePooling2D, Input, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
import time


img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


TRAINING_DIR = "dataSet\Train"

training_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)


VALIDATION_DIR = "dataSet\Test"

validation_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR,target_size=(224,224),batch_size = 10)


validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, target_size=(224,224),batch_size= 10)

input_tensor = Input(shape=(224, 224, 3))
x = img_augmentation(input_tensor)

model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

model.trainable = False

# Rebuild top
x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
x = layers.BatchNormalization()(x)

top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = layers.Dense(2, activation="softmax", name="pred")(x)




# Compile
model = tf.keras.Model(input_tensor, outputs, name="EfficientNet")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)


for layer in model.layers[-20:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

# model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
start = time.time()

history = model.fit(
train_generator,
# steps_per_epoch = 20,
epochs = 7,
validation_data = validation_generator,
verbose=1
# validation_steps = 14)
)

print("Total time for training: ",time.time()-start)

model.save('efficient_net', save_format='h5')

history_df = pd.DataFrame(history.history)
print(history_df)
history_df.plot()
plt.show()
plt.savefig("eff_net")









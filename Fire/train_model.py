# import tensorflow as tf
# import keras_preprocessing
# from keras_preprocessing import image
# from keras_preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# import pandas as pd
# import matplotlib.pyplot as plt
# # from tensorflow.keras.optimizers import Adam
# TRAINING_DIR = "Dataset\Train"
# training_datagen = ImageDataGenerator(rescale = 1./255,
#                                   horizontal_flip=True,
#                                   rotation_range=30,
#                                   height_shift_range=0.2,
#                                   fill_mode='nearest')
# VALIDATION_DIR = "Dataset\Validation"
# validation_datagen = ImageDataGenerator(rescale = 1./255)
# train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
#                                          target_size=(224,224),
#                                          class_mode='categorical',
#                                          batch_size = 64)
# validation_generator = validation_datagen.flow_from_directory(      
#                                            VALIDATION_DIR,
#                                            target_size=(224,224),
#                                            class_mode='categorical',
#                                            batch_size= 16)

# model = Sequential([
# tf.keras.layers.Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=(224, 224, 3)),
# tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2)),
# tf.keras.layers.Conv2D(256, (5,5), activation='relu'),
# tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2)),
# tf.keras.layers.Conv2D(384, (5,5), activation='relu'),
# tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2)),
# tf.keras.layers.Flatten(),
# tf.keras.layers.Dropout(0.2),
# tf.keras.layers.Dense(2048, activation='relu'),
# tf.keras.layers.Dropout(0.25),
# tf.keras.layers.Dense(1024, activation='relu'),
# tf.keras.layers.Dropout(0.2),
# tf.keras.layers.Dense(2, activation='softmax')])

# model.compile(loss='categorical_crossentropy',
# # optimizer=Adam(lr=0.0001),
# optimizer='adam',
# metrics=['acc'])

# history = model.fit(
# train_generator,
# steps_per_epoch = 15,
# epochs = 50,
# validation_data = validation_generator,
# validation_steps = 15
# )

# model.save('fire_detection.model', save_format='h5')

# history_df = pd.DataFrame(history.history)
# # print(history_df)
# history_df.plot()
# plt.show()



# # print(model.evaluate(X_test, y_test))

import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import SGD
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


TRAINING_DIR = "DataSet\Train"
training_datagen = ImageDataGenerator(rescale=1./255,
                                    zoom_range=0.15,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
VALIDATION_DIR = "DataSet\Test"
validation_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = training_datagen.flow_from_directory(
TRAINING_DIR,
target_size=(224,224),
shuffle = True,
class_mode='categorical',
batch_size = 64)

validation_generator = validation_datagen.flow_from_directory(
VALIDATION_DIR,
target_size=(224,224),
class_mode='categorical',
shuffle = True,
batch_size= 14)


input_tensor = Input(shape=(224, 224, 3))
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# for layer in base_model.layers:
#   layer.trainable = False

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# history = model.fit(
# train_generator,
# steps_per_epoch = 14,
# epochs = 20,
# validation_data = validation_generator,
# validation_steps = 14)
for layer in model.layers[:249]:
  layer.trainable = False
for layer in model.layers[249:]:
  layer.trainable = True#Recompile the model for these modifications to take effect
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(
train_generator,
steps_per_epoch = 14,
epochs = 16,
validation_data = validation_generator,
validation_steps = 14)

# model.save('fire_detection_1', save_format='h5')
model.save("fire_11.h5")

history_df = pd.DataFrame(history.history)
print(history_df)
history_df.plot()
plt.show()

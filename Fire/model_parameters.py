import tensorflow as tf

model = tf.keras.models.load_model('fire_detection_1')

print(model.summary())

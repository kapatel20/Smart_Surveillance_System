import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from keras.preprocessing import image
import matplotlib.image as mpimg
import numpy as np

# model = load_model('temp2')
model = load_model('efficient_net')

# face = cv2.CascadeClassifier('haar_face.xml')

cap = cv2.VideoCapture(0)

categories = {0: "Mask", 1: "No Mask"}
color = {0: (0,255,0), 1: (0,0,255)}

while True:
    tf, frame1 = cap.read()
    frame = cv2.resize(frame1, (224,224))
    frame = image.img_to_array(frame)
    frame = np.expand_dims(frame,axis=0)
    # frame = preprocess_input(frame)
    result = model.predict(frame)
    label = np.argmax(result)
    # print(categories[label])
    cv2.putText(frame1, categories[label], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.imshow('LIVE', frame1)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()








# tf, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:

#         # # face_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
#         # resized = cv2.resize(frame, (224, 224))
#         # normalized = resized/255.0
#         # # reshaped = np.reshape(normalized, (3, 200, 200, 1))
#         # result = model.predict(normalized)
#         im = Image.fromarray(frame, 'RGB')#Resizing into 224x224 because we trained the model with this image size.
#         im = im.resize((224,224))
#         img_array = image.img_to_array(im)
#         img_array = np.expand_dims(img_array, axis=0) / 255
#         result = model.predict(img_array)[0]

#         label = np.argmax(result)
#         print(categories[label])
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         # cv2.putText(frame, categories[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))
#         cv2.putText(frame, categories[label],(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_4)
#         cv2.imshow('LIVE', frame)
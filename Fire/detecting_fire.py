# from tensorflow.keras.models import load_model
# from keras.preprocessing import image
# import numpy as np

# # uploaded = files.upload()
# model = load_model('fire_detection.model')
# # for fn in uploaded.keys():
# path = 'Users\patel\Desktop\my\FireDetection\Prediction\1.jpg'
# img = image.load_img(path, target_size=(224,224))
# x = image.img_to_array(img)
# x=np.expand_dims(x,axis=0)/255
# classes = model.predict(x)
# print(np.argmax(classes[0]==0), max(classes[0]))

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image#Load the saved model
model = tf.keras.models.load_model('fire_detection_1')
categories = {0: "Fire", 1: "NoFire"}
video = cv2.VideoCapture(0)
while True:
    kp, frame = video.read()#Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')#Resizing into 224x224 because we trained the model with this image size.
    im = im.resize((224,224))
    img_array = image.img_to_array(im)
    img_array = np.expand_dims(img_array, axis=0) / 255
    probabilities = model.predict(img_array)[0]
    #Calling the predict method on model to predict 'fire' on the image
    prediction = np.argmax(probabilities)
    #if prediction is 0, which means there is fire in the frame.
    # if prediction == 0:
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    print(categories[prediction])
    cv2.putText(frame, categories[prediction],(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_4)
    # print(probabilities[prediction])
    cv2.imshow("Capturing", frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
import numpy as np
import cv2 as cv
import pickle
import tensorflow as tf
##########################
width = 640
height = 480

threshold = 0.8


###########################
cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# pickle_in = open("integer_detection_model.p", "rb")
# model = pickle.load(pickle_in)

model = tf.keras.models.load_model('my_model.h5')
# function for Pre Processing the image
def preProcessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOrgianl = cap.read()
    img = np.asarray(imgOrgianl)
    img = cv.resize(img, (32, 32))
    img = preProcessing(img)
    cv.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    # Predict
    classIndex = int(model.predict_classes(img))

    predictions = model.predict(img)
    # in the prediction array will gives the element that highest one
    probVal = np.amax(predictions)
    print(classIndex, probVal)

    if probVal > threshold:
        cv.putText(imgOrgianl, str(classIndex) + " " + str(probVal), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv.imshow("Orginal Image", imgOrgianl)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


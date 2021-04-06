import numpy as np
import cv2 as cv
import os

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

import pickle

import keras
##############################################
path = 'myData'
test_ratio = 0.2
Val_ratio = 0.2
imageDimention = (32, 32, 3)


batch_sizeVal = 50
epochsVal = 10
stepsPerEpoch = 2000

############################################

images = []
classNo = []
myList = os.listdir(path)
myList.remove('.DS_Store')
print("Total number of classes detected",len(myList))
num_classes = len(myList)
print("Importing Classes....")

for x in range(0, num_classes):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv.imread(path+"/"+str(x)+"/"+y)
        curImg = cv.resize(curImg, (32, 32))
        images.append(curImg)
        classNo.append(x)

    print(x, end=" ")

print(" ")

images = np.array(images)
classNo = np.array(classNo)

print("Images", images.shape)
print("class number", classNo.shape)

###########
#Spliting the Data training, testing, validation
# X_train contain actual images, y_train contain ID's of each images
###########
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=test_ratio)
X_train, X_validation, y_train, y_Validation = train_test_split(X_train, y_train, test_size=Val_ratio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

# to check the number of  images contain each classes
num_of_samples = []
for x in range(0, num_classes):
    #print("{x} class == ", len(np.where(y_train==x)[0]))
    num_of_samples.append(len(np.where(y_train==x)[0]))
print(num_of_samples)

# to plot the number of images for each class
# plt.figure(figsize=(10, 5))
# plt.bar(range(0, num_classes), num_of_samples)
# plt.title("number of images for each class")
# plt.xlabel("ClassID")
# plt.ylabel("Number Of Images")
# plt.show()

# function for Pre Processing the image
print("before the pre processing X_train images:", X_train.shape)
def preProcessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img/255
    return img

# to show the sample of preprocessed
# img = preProcessing(X_train[10])
# img = cv.resize(img, (300, 300))
# cv.imshow("pre process", img)
# cv.waitKey(0)

# Pre Process all the X_train images and convert into numpy array
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))
# img = X_train[10]
# img = cv.resize(img, (300, 300))
# cv.imshow("pre process", img)
# cv.waitKey(0)
print("after the pre processing X_train images:", X_train.shape)
print("after the pre processing X_test images:", X_test.shape)
print("after the pre processing X_validation images:", X_validation.shape)

# Create Depth one for all the images it require for the CNN run properly
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
print("After Adding Depth", X_train.shape)

# Augmenting the images
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,shear_range=0.1,rotation_range=10)
dataGen.fit(X_train)
# One hot encoding
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_Validation = to_categorical(y_Validation, num_classes)

# we are creating CNN model as LeNet
# for understanding architecture https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

def myModel():
    no_of_filter = 60
    size_filter1 = (5, 5)
    size_filter2 = (3, 3)
    size_pool = (2, 2)
    no_nodes = 500

    model = Sequential()
    model.add((Conv2D(no_of_filter, size_filter1, input_shape=(imageDimention[0], imageDimention[1], 1),
                      activation='relu')))
    model.add((Conv2D(no_of_filter, size_filter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add((Conv2D(no_of_filter//2, size_filter2, activation='relu')))
    model.add((Conv2D(no_of_filter//2, size_filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
model = myModel()
print(model.summary())

history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch_sizeVal),
                                 steps_per_epoch=stepsPerEpoch,epochs=epochsVal,
                    validation_data= (X_validation, y_Validation),
                    shuffle=1)

# plt.figure(1)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['training','validation'])
# plt.title('Loss')
# plt.xlabel('epoch')
#
# plt.figure(2)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.legend(['training','validation'])
# plt.title('Accuracy')
# plt.xlabel('epoch')
# plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])

model.save('my_model.h5')
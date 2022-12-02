import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
from tqdm import tqdm
#from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

DATADIR = 'Nonsegmented/'
#CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed',
#            'Shepherd’s Purse', 'Small-flowered Cranesbill', 'Sugar beet']

CATEGORIES = ['Common wheat', 'Sugar beet']

IMG_SIZE = 60

training_data = []

hm, HM, sm, SM, vm, VM = 29, 90 , 0, 255, 32, 122
lower = np.array([hm,sm,vm])
upper = np.array([HM,SM,VM])

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            
            img_array = cv2.imread(os.path.join(path,img))
            hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,lower,upper)
            filtered = cv2.bitwise_and(img_array,img_array,mask=mask)
            #gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
            new_array = cv2.resize(filtered, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])


create_training_data()

##################################

print(len(training_data))

random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3)

###############################################

# model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
# model.add(MaxPooling2D((2,2)))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D((2,2)))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# print('pre pre')

# model.fit(X,Y, epochs=10)

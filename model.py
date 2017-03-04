import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam

def sampling(data):
    st = data['steering'].astype(np.float32)
    bins, num = 100, 100
    c, b = np.histogram(st, bins)
    dist = np.digitize(st, b)
    return np.concatenate([data[dist==i][:num] for i in range(bins)])

def read_image(f):
    f = f.strip()
    image = cv2.cvtColor(cv2.imread(os.path.join('data/', f)), cv2.COLOR_BGR2YUV)
    return image 


data = pd.read_csv('data/driving_log.csv')
sampled_data = sampling(data)


center, left, right, steer, _, _, _ = np.split(sampled_data, 7, axis=1)
adjust_angle = 0.25

images_names = np.append(center, left)
images_names = np.append(images_names, right)

measurements = np.append(steer, steer + adjust_angle)
measurements = np.append(measurements, steer - adjust_angle)

print(images_names.shape, measurements.shape)


images = np.array([read_image(f) for f in images_names])


X_train = np.asarray(images)
y_train = np.asarray(measurements)


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
print(model.output_shape)

model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')


model.fit(X_train, y_train, validation_split=0.2,shuffle=True,nb_epoch=5)

model.save('model.h5')


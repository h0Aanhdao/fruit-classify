import os
import numpy as np
import matplotlib.image as mp_img
import random

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizer_v2.adam import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

train_path = "C:/Users/n/Desktop/fruit-classify/fruits/Training/"
test_path = "C:/Users/n/Desktop/fruit-classify/fruits/Test/"

categories_test = [name for name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, name))]
categories_train = [name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))]

all_array_test = []
all_array_train = []

x_train = []
y_train = []

x_test = []
y_test = []

data_generator = ImageDataGenerator(rescale=1. / 255)

train_generator = data_generator.flow_from_directory(train_path, target_size=(100, 100, 3),
                                                     batch_size=18, color_mode="rgb", class_mode="categorical")

test_generator = data_generator.flow_from_directory(test_path, target_size=(100, 100, 3),
                                                    batch_size=18, color_mode="rgb", class_mode="categorical")

for i in categories_train:
    path = os.path.join(train_path, i)
    class_num = categories_train.index(i)

    for img in os.listdir(path):
        img_array = mp_img.imread(os.path.join(path, img))

        all_array_train.append([img_array, class_num])

for j in categories_test:
    path = os.path.join(test_path, j)
    class_num = categories_test.index(j)

    for img in os.listdir(path):
        img_array = mp_img.imread(os.path.join(path, img))

        all_array_test.append([img_array, class_num])

fruits_array_train = []
for features, label in all_array_train:
    fruits_array_train.append(features)

# location = [[1, 500, 1150], [1500, 2000, 2500], [3000, 3500, 4000]]
# a = 0
# b = 1
# c = 2

# for i, j, k in location:
#     plt.subplots(figsize=(8, 8))
#     plt.subplot(1, 3, 1)
#     plt.imshow(fruits_array_train[i])
#     plt.title(categories_train[a])
#     plt.axis("off")
#     plt.subplot(1, 3, 2)
#     plt.imshow(fruits_array_train[j])
#     plt.title(categories_train[b])
#     plt.axis("off")
#     plt.subplot(1, 3, 3)
#     plt.imshow(fruits_array_train[k])
#     plt.title(categories_train[c])
#     plt.axis("off")
#     a += 3
#     b += 3
#     c += 3

# random.shuffle(all_array_train)

for features, label in all_array_train:
    x_train.append(features)
    y_train.append(label)

x_train = np.array(x_train)
random.shuffle(all_array_test)

for features, label in all_array_test:
    x_test.append(features)
    y_test.append(label)

x_test = np.array(x_test)

x_train = x_train.reshape(-1, 100, 100, 3)
x_train = x_train / 255

x_test = x_test.reshape(-1, 100, 100, 3)
x_test = x_test / 255

y_train = to_categorical(y_train, num_classes=len(categories_train))
y_test = to_categorical(y_test, num_classes=len(categories_test))

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

try:
    model = load_model('model.h5')
except IOError:
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(3, 3), padding="Same", activation="relu", input_shape=(100, 100, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.35))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(categories_train), activation="softmax"))

    # defining optimizer
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    # compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    data_gen = ImageDataGenerator(featurewise_center=False,
                                  samplewise_center=False,
                                  featurewise_std_normalization=False,
                                  samplewise_std_normalization=False,
                                  zca_whitening=False,
                                  rotation_range=0.5,
                                  zoom_range=0.5,
                                  width_shift_range=0.5,
                                  height_shift_range=0.5,
                                  horizontal_flip=False,
                                  vertical_flip=False)

    data_gen.fit(x_train)

    history = model.fit(data_gen.flow(x_train, y_train, batch_size=18), epochs=5,
                        validation_data=(x_val, y_val), steps_per_epoch=x_train.shape[0] // 18)

    model.save('model.h5')

# test sau khi chạy model
#path = load_img('fruits\\Training\\Apple Golden 1\\0_100.jpg', target_size=(100, 100, 3))

#plt.imshow(path)
#plt.show()

#print(np.argmax(model.predict(np.expand_dims(img_to_array(path), 0))))
#print(categories_train[np.argmax(model.predict(np.expand_dims(img_to_array(path), 0)))])
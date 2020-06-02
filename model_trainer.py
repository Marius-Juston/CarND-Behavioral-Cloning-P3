import csv
import os

import cv2
import numpy as np
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D
from tensorflow.python.keras.models import Model, Sequential


def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - .5, input_shape=(160, 320, 3)))

    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model


def save_model(model: Model, file_path):
    model.save(file_path, save_format='h5')


def load_images(driving_log_file):
    lines = []

    with open(driving_log_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)

        for line in reader:
            lines.append(line)

    images = []
    measurements = []

    separator = os.sep
    image_directory = 'data/IMG/'

    for line in lines:
        source_path = line[0]
        file_name = source_path.split(separator)[-1]
        path = image_directory + file_name

        image = cv2.imread(path)
        images.append(image)

        measurement = float(line[3])
        measurements.append(measurement)

    return np.array(images), np.array(measurements)


import matplotlib.pyplot as plt

if __name__ == '__main__':
    image_file = 'data/driving_log.csv'
    images, measurements = load_images(image_file)

    plt.imshow(images[0])
    plt.show()

    model = create_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit(images, measurements, shuffle=True, validation_split=.2, epochs=12)

    model.save('model.h5')

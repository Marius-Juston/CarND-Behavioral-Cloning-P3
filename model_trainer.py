import csv
import os
from datetime import datetime

import cv2
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from tensorflow.python.keras.models import Model, Sequential


def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - .5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 2, activation='relu'))
    model.add(Convolution2D(36, 5, 2, activation='relu'))
    model.add(Convolution2D(48, 5, 2, activation='relu'))
    model.add(Convolution2D(64, 3, activation='relu'))
    model.add(Convolution2D(64, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model


def save_model(model: Model, file_path):
    model.save(file_path, save_format='h5')


def get_image(line, index, array, separator=os.sep, image_directory='data/IMG/'):

    source_path = line[index]
    file_name = source_path.split(separator)[-1]
    path = image_directory + file_name
    image = cv2.imread(path)
    if image is None:
        print(index, line)
    array.append(image)
    array.append(cv2.flip(image, 1))


def load_images(driving_log_file):
    lines = []

    with open(driving_log_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)

        for line in reader:
            lines.append(line)

    center = []
    left = []
    right = []
    measurements = []

    for line in lines:
        get_image(line, 0, center)
        get_image(line, 1, left)
        get_image(line, 2, right)

        measurement = float(line[3])
        measurements.append(measurement)
        measurements.append(-measurement)

    return np.array(center), np.array(left), np.array(right), np.array(measurements)


if __name__ == '__main__':
    image_file = 'data/driving_log.csv'
    save_images = 'images.npz'

    if os.path.exists(save_images):
        print("Loading")
        data = np.load(save_images, allow_pickle=True)
        images, measurements, left, right = data['images'], data['measurements'], data['left'], data['right']
    else:
        images, left, right, measurements = load_images(image_file)
        np.savez(save_images, images=images, measurements=measurements, left=left, right=right)

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=5)

    model = create_model()
    model.summary()

    model.compile(loss='mse', optimizer='adam')
    model.fit(images, measurements, shuffle=True, validation_split=.2, epochs=12, callbacks=[early_stop])

    model.save(f'model{datetime.now().strftime("%Y%m%d-%H%M%S")}.h5')

import csv
import os
from datetime import datetime
from math import ceil

import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, BatchNormalization
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.training.adam import AdamOptimizer


def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - .5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 2, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Convolution2D(36, 5, 2, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Convolution2D(48, 5, 2, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1164, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())

    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
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


def generator(samples, driving_log_file, batch_size=64):
    num_samples = len(samples)
    image_dir = driving_log_file.split('/')[0] + "/IMG/"

    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = image_dir + batch_sample[0].split(os.sep)[-1]
                center_image = cv2.imread(name)
                center_angle = batch_sample[1]

                if batch_sample[2]:
                    center_image = cv2.flip(center_image, 1)

                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            yield tuple(sklearn.utils.shuffle(X_train, y_train))


def load_images(driving_log_file, batch_size=64, correction=.2):
    lines = []

    with open(driving_log_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)

        for line in reader:
            steering_center = float(line[3])
            steering_right = steering_center - correction
            steering_left = steering_center + correction

            lines.append((line[0], steering_center, 0))
            lines.append((line[0], steering_center, 1))
            lines.append((line[1], steering_left, 0))
            lines.append((line[1], steering_left, 1))
            lines.append((line[2], steering_right, 0))
            lines.append((line[2], steering_right, 1))

    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, driving_log_file, batch_size=batch_size)
    validation_generator = generator(validation_samples, driving_log_file, batch_size=batch_size)

    return train_generator, validation_generator, len(train_samples), len(validation_samples)


if __name__ == '__main__':
    image_file = 'data/driving_log.csv'

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=5)

    model = create_model()
    model.summary()

    model.compile(loss='mse', optimizer=AdamOptimizer(0.0001))

    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    logdir = "logs/fit/" + date_time
    tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

    batch_size = 64

    train_generator, validation_generator, l_train_samples, l_validation_samples = load_images(image_file, batch_size)
    print(l_train_samples, l_validation_samples)

    model.fit(train_generator,
              steps_per_epoch=ceil(l_train_samples / batch_size),
              validation_data=validation_generator,
              validation_steps=ceil(l_validation_samples / batch_size),
              epochs=5, verbose=1, callbacks=[early_stop, tensorboard_callback])

    model.save(f'model{date_time}.h5')

# ch03
# The Boston Housing Price dataset (p.92)
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


# Preprocessing
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# Building our network (p.93)
# In general, the less training data you have, the worse overfitting will be.
# Using a small network is one way to mitigate overfitting.
from keras import models
from keras import layers


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))  # No activation function required for the output layer
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# Validating our approach using k-fold cross-validation (p.94)
import numpy as np

k = 4
num_val_samples = len(train_data) // k
all_scores = []
for i in range(k):
    print('processing fold #', i)
    # prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
        axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[: i * num_val_samples], train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )

    # build the Keras model (already compiled)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=100, batch_size=1)
    # evaluate the model on the validation data
    val_mse, val_mae = model.evaluate(val_data, val_targets)
    all_scores.append(val_mae)

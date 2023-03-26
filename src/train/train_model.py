import numpy as np
import tensorflow as tf
from tensorflow import keras


# train model
def train_model(model, dataset):

    X_train, y_train, X_val, y_val, X_test, y_test = dataset

    hobj = model.fit(X_train, y_train, validation_data=(
        X_val, y_val), epochs=100, batch_size=200, shuffle=True)

    return model

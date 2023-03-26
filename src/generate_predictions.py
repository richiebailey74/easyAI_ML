import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def predict_model(model, X_test): # might need different params (for accessing preprocessor and model)

    return model.predict(X_test)

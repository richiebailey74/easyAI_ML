import numpy as np
import pandas as pd
import copy
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
import sys

from create_workspace_object import *

fileName = sys.argv[1]
nn_type = sys.argv[2]

WS_obj = create_workspace_object(fileName, nn_type, False)

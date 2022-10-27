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

from model.generate_nn_model import *
from model.generate_nn_params import *
from model.train_model import *
from preprocess.preprocess_data import *
from preprocess.train_val_test_split import *
from eval_model import *
from workspace.labelize_softmax import *
from workspace.predict_model import *
from workspace.WorkspaceObject import *

# params: dataset, attribute values of all the dropdowns (is image data, orientation of inputted csv)
# nn_type can be: 'classification-binary', 'classification-multi', 'regression' (can be extended later)


def create_workspace_object(data_name, nn_type, isIncorrectOrientation):

    # read in data
    data = pd.read_csv(f'data/{data_name}.csv')

    # preprocess data method here
    targets, data_preprocessed = preprocess_data(data, isIncorrectOrientation)

    # get dataset splits
    dataset = train_val_test_split(data_preprocessed, targets, nn_type)

    # generate params
    nn_params = generate_nn_params(dataset, nn_type)

    # generate nn model
    model = generate_nn_model(dataset, nn_params)

    # train model and save
    model_fitted = train_model(model, dataset)
    model_fitted.save(f'model/models/{data_name}_model')

    # model predictions and save them
    preds = predict_model(model_fitted, dataset)
    if nn_type == 'classification':
        preds = labelize_softmax(preds)
    pd.DataFrame(preds).to_csv(
        f'workspace/predictions/{data_name}_predictions.csv')

    # eval model
    performance_metric = eval_model(preds, dataset, nn_type)

    # create object here
    workspace_object = WorkspaceObject(model_fitted, performance_metric)

    # return object (or do whatever needs to be done)
    return workspace_object

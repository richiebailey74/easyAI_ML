import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# TODO: will eventually contain functionality to deal with varying data types and types of data
def preprocess_data(data_raw, transposed):
        
    # transposed will be true if each column is a sample instead of each row (just flip it)
    if transposed == True:
        data_raw = data_raw.T
        
    # remove target values
    targets = data_raw.iloc[:,-1]
    data_raw.drop(data_raw.columns[-1], axis=1)
        
    # nan handling: remove feature if over 40% of column is nan
    to_drop = []
    for col_name in data_raw:
        if data_raw[col_name].isna().sum() / len(data_raw.axes[0]) > .4:
            to_drop.append(col_name)
        if data_raw[col_name].dtype == 'str' or data_raw[col_name].dtype == 'datetime64':
            to_drop.append(col_name)
    
    data_raw.drop(to_drop, axis=1)
    
    # nan handling: KNN impute the remaining nans with n_neighbors as a function of the number of
    knn_imputer = KNNImputer(n_neighbors=(.05 * len(data_raw.axes[0])), weights='uniform', metric='nan_euclidean')
    data_imputed = knn_imputer.fit_transform(data_raw)
        
    # standardize each column respective to itself (not normalizing because we aren't removing outliers)
    data_return = np.copy(data_imputed.T)
    for ind, col in enumerate(data_imputed.T):
        temp = StandardScaler().fit_transform(col.reshape(-1,1))
        data_return[ind] = temp.ravel()
        
    return np.array(targets), data_return.T
    
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import numpy as np

# 1차원 행별로 가우지안스무딩
def gaussianSmoothing(data, std):
    data_temp = data.copy()
    X_data = data_temp.values
    for i in range(X_data.shape[0]):
        temp = X_data[i, :]
        temp_gaussian = gaussian_filter1d(temp, std)
        if i == 0:
            output = temp_gaussian
        else:
            output = np.vstack((output, temp_gaussian))
    return pd.DataFrame(output)    
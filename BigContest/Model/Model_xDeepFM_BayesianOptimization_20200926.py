import warnings
warnings.filterwarnings(action='ignore')

import os
os.chdir('/content/drive/My Drive/Big-Contest')

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import KernelPCA, PCA

# from deepctr.models import DeepFM
from deepctr.models.xdeepfm import xDeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from bayes_opt import BayesianOptimization

import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from functools import partial

# 데이터 로드
# data = pd.read_csv("./data/범주데이터모음.csv", usecols=lambda x: 'Unnamed' not in x)
# data = pd.read_csv("/content/drive/Shared drives/빅콘테스트/데이터(전처리 후 논의 대상)/범주데이터모음2.csv", 
#                    usecols=lambda x: 'Unnamed' not in x)
# data2 = pd.read_csv("/content/drive/Shared drives/빅콘테스트/데이터(전처리 후 논의 대상)/deepfm데이터.csv",
#                     usecols=lambda x: 'Unnamed' not in x)
# data = pd.read_csv("./data/input(csv).csv", usecols=lambda x:'Unnamed' not in x)
# data = pd.read_excel("/content/drive/Shared drives/빅콘테스트/2020빅콘테스트 문제데이터(데이터분석분야-챔피언리그)/01_제공데이터/input_data(raw).xlsx",
#                      usecols=lambda x:'Unnamed' not in x)
data = pd.read_csv("./data/input(raw)_3.csv", usecols=lambda x:'Unnamed' not in x)
data

# 컬럼 확인
len(data.columns), data.columns

# 취급액 0인 데이터 제외
df = data.copy()
df = df.loc[df['AMT'] != 0].reset_index(drop=True)
df

# 판매단가 minmax scaling
ms = MinMaxScaler()
df['price'] = ms.fit_transform(df[['price']])
# df['productprice'] = np.log1p(df['productprice'])
df

# 컬럼 변경
df = df.drop(columns='year', axis=1)
df

# feature, target 정의     
sparse_features = ['month', 'day', 'day_of_week', 'hour', 'minute', 'is_continuous', 'pcategory', 'pname',
                   'warning', 'is_warning', 'warning_bin4', 'warning_bin3', 'alert', 'is_alert', 
                   'daily_temperature_mean_1', 'daily_temperature_max_1', 'daily_temperature_min_1', 'daily_temperature_diff_1', 'daily_temperature_mean_2', 'daily_temperature_max_2', 'daily_temperature_min_2', 'daily_humidity_mean_1', 'daily_humidity_mean_2', 'daily_humidity_max_2', 'daily_sunshine_1', 
                   'gonghang_20_cat', 'pildong_40_cat', 'gonghang_50_cat', 'jongro56_30_cat', 'samcheong_40_cat', 'jongro56_40_cat', 'gonghang_40_cat', 'jegi_40_cat', 'pildong_30_cat', 'sogong_20_cat', 'samcheong_50_cat', 'sogong_30_cat', 'gonghang_30_cat',  'sogong_40_cat']
dense_features = ['price', 'exposure', 
                  'exposure_cnt_1', 'exposure_cnt_2', 'exposure_cnt_3', 'exposure_cnt_4',  'exposure_cnt_5', 'exposure_cnt_6', 'exposure_cnt_7', 'exposure_cnt_8', 'exposure_cnt_9', 'exposure_cnt_10', 'exposure_cnt_11', 'exposure_cnt_12', 'exposure_cnt_sum',
                  'dust_seoul', 'dust_busan', 'dust_daegu', 'dust_incheon', 'dust_gwangju', 'dust_daejeon', 'dust_ulsan', 'dust_gyeonggi', 'dust_gangwon', 'dust_chungbuk', 'dust_chungnam', 'dust_cheonbuk', 'dust_cheonnam', 'dust_gyeongbuk', 'dust_gyeongnam', 'dust_sejong',
                  'daily_temperature_mean', 'daily_temperature_max', 'daily_temperature_min', 'daily_temperature_diff', 'daily_precipitation_sum', 'daily_humidity_mean', 'daily_humidity_max', 'daily_sunshine_sum', 'daily_sunshine_max', 'daily_insolation_sum', 'daily_snow_sum',
                  'gonghang_20', 'pildong_40', 'gonghang_50', 'jongro56_30', 'samcheong_40', 'jongro56_40', 'gonghang_40', 'jegi_40', 'pildong_30', 'sogong_20', 'samcheong_50', 'sogong_30', 'gonghang_30', 'sogong_40']
target = ['AMT']

# feature column 정의
fixlen_feature_columns = [SparseFeat(feat, df[feat].nunique(), embedding_dim=6) for feat in sparse_features] + \
                         [DenseFeat(feat, 1, ) for feat in dense_features]                                                         
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print(fixlen_feature_columns)
print(feature_names)

# 트레인, 테스트 셋 분리
X_train, X_test, y_train, y_test = train_test_split(df[feature_names], df[target],
                                                    test_size=0.2, random_state=2020)

# 입력 데이터
X_train = {name:X_train[name].values for name in feature_names}
X_test = {name:X_test[name].values for name in feature_names}

def train_and_validate(linear_feature_columns, dnn_feature_columns, train_ds, train_label, test_ds, test_label, **kwargs):

    dnn_hidden_units = kwargs.pop('dnn_hiden_units', 256)
    cin_layer_size = kwargs.pop('cin_layer_size', 128)
    cin_split_half = kwargs.pop('cin_split_half', True)
    cin_activation = kwargs.pop('cin_activtion', 'relu')
    l2_reg_linear = kwargs.pop('l2_reg_linear', 1e-05)
    l2_reg_embedding = kwargs.pop('l2_reg_embedding', 1e-05)
    l2_reg_dnn = kwargs.pop('l2_reg_dnn', 0)
    l2_reg_cin = kwargs.pop('l2_reg_cin', 0)
    seed = kwargs.pop('seed', 1024)
    dnn_dropout = kwargs.pop('dnn_dropout', 0)
    dnn_activation = kwargs.pop('dnn_activation', 'relu')
    dnn_use_bn = kwargs.pop('dnn_use_bn', False)
    task = kwargs.pop('task', 'regression')

    K.clear_session()

    model = xDeepFM(linear_feature_columns, dnn_feature_columns,
                    dnn_hidden_units = (dnn_hidden_units, dnn_hidden_units),
                    cin_layer_size = (cin_layer_size, cin_layer_size),
                    cin_split_half = cin_split_half,
                    cin_activation = cin_activation,
                    l2_reg_linear = l2_reg_linear,
                    l2_reg_embedding = l2_reg_embedding,
                    l2_reg_dnn = l2_reg_dnn,
                    l2_reg_cin = l2_reg_cin,
                    seed = seed,
                    dnn_dropout = dnn_dropout,
                    dnn_activation = dnn_activation,
                    dnn_use_bn = dnn_use_bn,
                    task = task)
    
    learning_rate = kwargs.pop('lr', 0.001)
    num_epochs = kwargs.pop('epochs', 300)
    batch_size = kwargs.pop('batch_size', 256)    

    model.compile(loss='mape', optimizer=optimizers.Adam(lr=learning_rate))

    hist = model.fit(train_ds, train_label,
                     batch_size=batch_size,
                     verbose=1,
                     validation_split=0.2,
                     shuffle=True)

    # 평가지표 산식
    def __mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    test_pred = model.predict(test_ds)
    score = __mean_absolute_percentage_error(test_label['AMT'], test_pred.reshape(-1, ))
    print("Test Set MAPE Score: ", score)

    return score

# partial
train_and_validate_partial = partial(train_and_validate, linear_feature_columns, dnn_feature_columns, X_train, y_train, X_test, y_test)

# 베이지안 optimization
pbounds = {'lr': (1e-4, 1e-2), 'l2_reg_linear': (1e-05, 1e-2)}
optimizer = BayesianOptimization(
    f = train_and_validate_partial,
    pbounds = pbounds,
    verbose=2,
    random_state=1024
)

optimizer.maximize(init_points=10, n_iter=10,)
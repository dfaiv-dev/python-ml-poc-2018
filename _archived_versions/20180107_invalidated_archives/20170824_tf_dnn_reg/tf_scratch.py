import pandas as pd
import tensorflow as tf
import tensorflow.contrib.learn as tf_learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_scripts import pcs_data_loader

tf.logging.set_verbosity(tf.logging.INFO)

df = pcs_data_loader.shape_pps_data(pcs_data_loader.load_corn_rows_mssql())
df.drop(['Area'], axis=1, inplace=True)
y = df['Dry_Yield']
X = df.drop(['Dry_Yield'], axis=1)
X_train, X_validation, y_train, y_validation = \
    train_test_split(X, y, test_size=.3, random_state=7)

scaler = StandardScaler()
scaler.fit(X)

feature_cols = [tf.feature_column.numeric_column(c) for c in X.columns.tolist()]
label = 'Dry_Yield'

nn = tf_learn.DNNRegressor(
    (1024, 512, 256), feature_cols,
    model_dir='./results/20170824_tf_dnn_reg/tf_dat')


def train_input_fn():
    return tf.estimator.inputs.pandas_input_fn(
        pd.DataFrame(scaler.transform(X_train), columns=X.columns.tolist()),
        y_train,
        shuffle=False
    )


nn.fit(input_fn=train_input_fn(), steps=5000)

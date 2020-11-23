import numpy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from data_scripts import pcs_data_loader
# load dataset
from modeling import score_util

df = pcs_data_loader.shape_pps_data(pcs_data_loader.load_corn_rows_mssql())
df.drop(['Area'], axis=1, inplace=True)
y = df['Dry_Yield']
X = df.drop(['Dry_Yield'], axis=1)
X_train, X_validation, y_train, y_validation = \
    train_test_split(X, y, test_size=.3, random_state=7)
scaler = StandardScaler()
scaler.fit(X_train)

# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dropout(.1, input_shape=(len(X.columns),), seed=91))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(.1, seed=71))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(.1, seed=51))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(.1, seed=31))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(.1, seed=11))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=10000, batch_size=17000, verbose=1)
kfold = KFold(n_splits=10, random_state=seed)
estimator.fit(scaler.transform(X_train), y_train.values,
              callbacks=[
                  ModelCheckpoint('./results/20170824_tf_dnn_reg/keras/val_acc_best.chkpnt', monitor="loss",
                                  save_best_only=True, save_weights_only=False, verbose=5),
                  EarlyStopping(monitor='loss', min_delta=.0001, patience=25, verbose=5)
              ])

estimator.model.load_weights('./results/20170824_tf_dnn_reg/keras/val_acc_best.chkpnt')
scr = score_util.score(estimator, scaler, X_validation, y_validation)
print(scr)
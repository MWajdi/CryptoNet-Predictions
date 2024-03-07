from models import *
import pandas as pd
from pandasql import sqldf
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Evaluating CNN+LSTM model

# Data prep
series = pd.read_csv("series.csv")
split = int(len(series) * 0.75)
s_train = series[:split]
s_valid = series[split:]

# Create windows and batchify

window_size = 30
batch_size = 32
shuffle_buffer = 1000

def windowed_dataset(dataframe, window_size, batch_size, shuffle=False, shuffle_buffer=0):
    series = dataframe["sales"].values
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    if shuffle:
        dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

s_train_windowed = windowed_dataset(s_train, window_size, batch_size)

# Tweak learning rate

def tweak_lr(data, model):
    lr_scheduler = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10 ** (epoch / 20))
    model.compile(optimizer='adam', loss=keras.losses.Huber())
    history = model.fit(data, epochs=100, callbacks=[lr_scheduler])
    lr = 1e-8 * 10 ** (np.arange(100) / 20)

    plt.semilogx(lr, history.history['loss'])
    plt.show()

# Fit model
fit_model = False

if fit_model:
    epochs = 100
    cnn_lstm_model.compile(optimizer='adam', loss='mse')
    history = cnn_lstm_model.fit(s_train_windowed, epochs=epochs)
    plt.plot(np.arange(100), history.history['loss'])
    plt.show()
    cnn_lstm_model.save(f"last_saved_model".epochs)
else:
    model = keras.models.load_model("post100epochs")

# Model evaluation

def dl_predict(data, model, window_size, batch_size):
    dataset = data["sales"].values
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    predictions = model.predict(dataset, verbose=0)
    return predictions


# On training data

print("____________________________________________________________________________\nFor Deep Learning algorithms:\n")

def dl_compare(data, model, window_size, set_nature):
    y_pred = dl_predict(data, model, window_size, 32)[:-1]
    y_true = data["sales"][window_size:].values
    time = data["date"][window_size:]

    plt.figure(figsize=(10, 6))
    plt.plot(time, y_pred, label='Predicted')
    plt.plot(time, y_true, label='Actual')
    plt.legend()
    plt.title("Deep Learning Algorithm")
    plt.show()

    print(f"MSE on {0} set:".format(set_nature), mean_squared_error(y_pred, y_true))
    print(f"MAE on {0} set".format(set_nature), mean_absolute_error(y_true,y_pred))


print("On training set: ")
dl_compare(s_train, model, window_size,"training")

# On test data

print("On validation set: ")
dl_compare(s_valid, model, window_size, "validation")

# Evaluating decision trees
print("____________________________________________________________________________\nFor Decision Trees:\n")

def rolling_windows(data, window_size):
    dataset = data["sales"].to_numpy().reshape(-1,1)
    x_train, y_train = [], []
    for i in range(len(dataset) - window_size):
        x_train.append(dataset[i:i+window_size].flatten())
        y_train.append(dataset[i+window_size])

    x_train = np.array(x_train)
    y_train = np.array(y_train).ravel()
    return x_train, y_train


def tree_compare(time, y_pred, y_true, model_nature):
    plt.figure(figsize=(10, 6))
    plt.plot(time, y_pred, label='Predicted')
    plt.plot(time, y_true, label='Actual')
    plt.title(model_nature)
    plt.legend()
    plt.show()

    print(f"MSE = ", mean_squared_error(y_pred, y_true))
    print(f"MAE = ", mean_absolute_error(y_true,y_pred))

x_train, y_train = rolling_windows(s_train, window_size)
t_train = s_train["date"][window_size:]
x_valid, y_valid = rolling_windows(s_valid, window_size)
t_valid = s_valid["date"][window_size:]

model = DecisionTreeRegressor(random_state=42)
model.fit(x_train, y_train)

# On training set
print("On training set: ")
y_pred = model.predict(x_train)
tree_compare(t_train, y_pred, y_train, "Decision Tree")

# On validation set
print("On validation set: ")
y_pred = model.predict(x_valid)
tree_compare(t_valid, y_pred, y_valid, "Decision Tree")

# Evaluating Random Forests
print("____________________________________________________________________________\nFor Random Forests:\n")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# On training set
print("On training set: ")
y_pred = model.predict(x_train)
tree_compare(t_train, y_pred, y_train, "Random Forests")

# On validation set
print("On validation set: ")
y_pred = model.predict(x_valid)
tree_compare(t_valid, y_pred, y_valid, "Random Forests")

# Evaluating XGBoost
print("____________________________________________________________________________\nFor XGBoost:\n")
model = XGBRegressor(objective="reg:squarederror", n_estimators=100, seed=42)
model.fit(x_train, y_train)

# On training set
print("On training set: ")
y_pred = model.predict(x_train)
tree_compare(t_train, y_pred, y_train, "XGBoost")

# On validation set
print("On validation set: ")
y_pred = model.predict(x_valid)
tree_compare(t_valid, y_pred, y_valid, "XGBoost")






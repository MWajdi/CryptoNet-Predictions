import tensorflow as tf
import keras
import numpy as np
from plot_utils import plot_series
from data_prep import df
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

# Import data, normalize it and convert it to a numpy array

scaler = MinMaxScaler(feature_range=(0,100))
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

series = df['Low'].to_numpy()
time = np.array(range(len(series)))

# Truncate series to reduce execution time 
series = series[-10000:] 
time = time[-10000:]


# Split dataset into training and validation sets

split_time = int(len(time) * 0.9)
x_train = series[:split_time]
train_time = time[:split_time]
x_valid = series[split_time:]
valid_time = time[split_time:]

print("x_valid: ", len(x_valid))
print("valid_time: ", len(valid_time))

# Prepare features and labels

def windowed_dataset(dataset, window_size, batch_size, shuffle_buffer):
    # Convert to tensor
    dataset = tf.data.Dataset.from_tensor_slices(dataset)

    # Create windows
    dataset = dataset.window(window_size+1, 1, drop_remainder=True)

    # Flatten windows
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))

    # Create labels
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Create batches
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

# Parameters
window_size = 30
batch_size = 32
shuffle_buffer = 1000

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer)


# Build the model
model = keras.Sequential([
    keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='causal', activation='relu', input_shape=[window_size,1]),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(64),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1)
])

# Tweak learning rate


def tweak_learning_rate(dataset, model):
    lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))

    model.compile(optimizer = keras.optimizers.SGD(momentum=0.9), loss=keras.losses.Huber())

    history = model.fit(dataset, epochs=100, verbose=2, callbacks=[lr_schedule])

    lrs = 1e-8 * 100**(np.arange(100) / 20)
    plt.semilogx(lrs, history.history['loss'])
    plt.show()

# tweak_learning_rate(train_set, model)
    
model.compile(optimizer = keras.optimizers.SGD(momentum=0.9, learning_rate=0.78), metrics=['mae'], loss=keras.losses.Huber())
history = model.fit(train_set, epochs=100, verbose=2 )


def model_forecast(model, series, window_size, batch_size):

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    
    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)
    
    # Get predictions on the entire dataset
    forecast = model.predict(dataset)
    
    return forecast

forecast_series = series[split_time-window_size + 1:]
forecast = model_forecast(model, forecast_series, window_size, batch_size)
result = forecast.squeeze()

print("result: ", len(result))

plot_series(valid_time, (result, x_valid))
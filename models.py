import tensorflow as tf
import keras

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

window_size = 30

cnn_lstm_model = keras.Sequential([
    keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='causal', activation='relu', input_shape=[window_size,1]),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(64),
    keras.layers.Dense(30),
    keras.layers.Dense(1)
])


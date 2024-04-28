import os 
import tensorflow as tf
#from tensorflow.keras.models import Sequential 
#from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

def load_model() -> tf.keras.Sequential: 
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool3D((1,2,2)))

    model.add(tf.keras.layers.Conv3D(256, 3, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool3D((1,2,2)))

    model.add(tf.keras.layers.Conv3D(75, 3, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool3D((1,2,2)))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(tf.keras.layers.Dropout(.5))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(tf.keras.layers.Dropout(.5))

    model.add(tf.keras.layers.Dense(41, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights(os.path.join('models','checkpoint'))

    return model

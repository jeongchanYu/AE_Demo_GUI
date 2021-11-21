import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, PReLU, Dense, Flatten, Reshape
import custom_function as cf
import numpy as np

class Encoder(tf.keras.Model):
    def __init__(self, default_float='float32'):
        super(Encoder, self).__init__()
        tf.keras.backend.set_floatx(default_float)
        self.encoder = []
        self.encoder.append(Conv1D(64, 21, strides=1, padding='same', activation='relu'))
        self.encoder.append(Conv1D(64, 21, strides=2, padding='same', activation='relu'))
        self.encoder.append(Conv1D(64, 21, strides=1, padding='same', activation='relu'))
        self.encoder.append(Conv1D(64, 21, strides=2, padding='same', activation='relu'))
        self.encoder.append(Conv1D(64, 21, strides=1, padding='same', activation='relu'))
        self.encoder.append(Conv1D(64, 21, strides=2, padding='same', activation='relu'))
        self.encoder.append(Conv1D(64, 21, strides=1, padding='same', activation='relu'))
        self.encoder.append(Conv1D(64, 21, strides=2, padding='same', activation='relu'))
        self.encoder.append(Conv1D(64, 21, strides=1, padding='same', activation='relu'))
        self.encoder.append(Conv1D(8, 21, strides=2, padding='same', activation='relu'))

    def call(self, x):
        output = x
        for f in self.encoder:
            output = f(output)
        return output


class Decoder(tf.keras.Model):
    def __init__(self, frame_size, default_float='float32'):
        super(Decoder, self).__init__()
        tf.keras.backend.set_floatx(default_float)
        self.frame_size = frame_size
        self.decoder = []
        # self.decoder.append(Flatten())
        # self.decoder.append(Dense(frame_size * 1024, activation='relu'))
        # self.decoder.append(Reshape([frame_size, 1024]))
        # self.decoder.append(Conv1D(1, self.frame_size//4, strides=1, padding='same', use_bias=False, kernel_initializer='zeros'))
        self.decoder.append(Conv1DTranspose(64, 21, strides=2, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1D(64, 21, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1DTranspose(64, 21, strides=2, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1D(64, 21, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1DTranspose(64, 21, strides=2, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1D(64, 21, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1DTranspose(64, 21, strides=2, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1D(64, 21, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1DTranspose(64, 21, strides=2, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1D(64, 21, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1DTranspose(64, 21, strides=2, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1D(64, 21, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1D(64, 21, strides=2, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1D(64, 21, padding='same'))
        self.decoder.append(PReLU())
        self.decoder.append(Conv1D(1, 21, padding='same', activation='tanh'))

        # self.noise_seed = tf.zeros([1, frame_size, 1])
        # self.noise_gen = tf.keras.layers.GaussianNoise(1.)
        # self.conv_noise_filter = Conv1D(256, 32, padding='same', activation='tanh')
        # self.dense_noise_select = Dense(256, activation='relu')

    def call(self, x):
        output = x
        for f in self.decoder:
            output = f(output)

        # noise = tf.tile(self.noise_seed, [x.shape[0], 1, 1])
        # noise = self.noise_gen(noise, True)
        # noise_filtered = self.conv_noise_filter(noise)
        # noise_select = tf.expand_dims(self.dense_noise_select(Flatten()(x)), 1)
        # noise_generated = tf.reduce_sum(noise_filtered * noise_select, 2, keepdims=True)
        #
        #
        # output += noise_generated
        return output


class Filter(tf.keras.Model):
    def __init__(self, frame_size, cut_off_freq, filter_order, sampling_freq, init=True, default_float='float32'):
        super(Filter, self).__init__()
        tf.keras.backend.set_floatx(default_float)
        self.frame_size = frame_size
        self.number_of_bands = len(cut_off_freq) + 1
        _, td_filter = cf.butterworth_filter(frame_size, cut_off_freq, filter_order,sampling_freq, default_float)
        circular_filter = np.pad(td_filter, [[0,0], [frame_size-1, 0]], "wrap")
        flip_td_filter = np.expand_dims(np.transpose(circular_filter), 1)
        if init:
            self.W = tf.Variable(flip_td_filter)
        else:
            self.W = tf.Variable(tf.random.truncated_normal(shape=[frame_size, 1, self.number_of_bands]))
        self.b = tf.Variable(tf.zeros(shape=[self.number_of_bands]))


    def call(self, x):
        return tf.nn.conv1d(x, self.W, stride=1, padding='SAME') + self.b

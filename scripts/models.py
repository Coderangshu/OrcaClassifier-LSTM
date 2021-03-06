from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import kapre
from kapre.composed import get_melspectrogram_layer
import tensorflow as tf
import os
import argparse

class CustomModel:
    def __init__(self, N_CLASSES=2, SR=20000, DT=2.0):
        self.N_CLASSES = N_CLASSES
        self.SR = SR
        self.DT = DT
        self.model = None

    def Conv1D(self):
        input_shape = (int(self.SR*self.DT), 1)
        i = get_melspectrogram_layer(input_shape=input_shape,
                                    n_mels=128,
                                    pad_end=True,
                                    n_fft=512,
                                    win_length=400,
                                    hop_length=160,
                                    sample_rate=self.SR,
                                    return_decibel=True,
                                    input_data_format='channels_last',
                                    output_data_format='channels_last')
        x = LayerNormalization(axis=2, name='batch_norm')(i.output)
        x = TimeDistributed(layers.Conv1D(8, kernel_size=(4), activation='tanh'), name='td_conv_1d_tanh')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_1')(x)
        x = TimeDistributed(layers.Conv1D(16, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_1')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_2')(x)
        x = TimeDistributed(layers.Conv1D(32, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_2')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_3')(x)
        x = TimeDistributed(layers.Conv1D(64, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_3')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_4')(x)
        x = TimeDistributed(layers.Conv1D(128, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_4')(x)
        x = layers.GlobalMaxPooling2D(name='global_max_pooling_2d')(x)
        x = layers.Dropout(rate=0.1, name='dropout')(x)
        x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
        if self.N_CLASSES==2:
            o = layers.Dense(1, activation='sigmoid', name='sigmoid')(x)
            self.model = Model(inputs=i.input, outputs=o, name='1d_convolution')
            self.model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        else:
            o = layers.Dense(self.N_CLASSES, activation='softmax', name='softmax')(x)
            self.model = Model(inputs=i.input, outputs=o, name='1d_convolution')
            self.model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        
        return self.model

    def Conv2D(self):
        input_shape = (int(self.SR*self.DT), 1)
        i = get_melspectrogram_layer(input_shape=input_shape,
                                    n_mels=128,
                                    pad_end=True,
                                    n_fft=512,
                                    win_length=400,
                                    hop_length=160,
                                    sample_rate=self.SR,
                                    return_decibel=True,
                                    input_data_format='channels_last',
                                    output_data_format='channels_last')
        x = LayerNormalization(axis=2, name='batch_norm')(i.output)
        x = layers.Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same', name='conv2d_tanh')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_1')(x)
        x = layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_1')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_2')(x)
        x = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_2')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
        x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
        x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dropout(rate=0.2, name='dropout')(x)
        x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
        if self.N_CLASSES==2:
            o = layers.Dense(1, activation='sigmoid', name='sigmoid')(x)
            self.model = Model(inputs=i.input, outputs=o, name='2d_convolution')
            self.model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        else:
            o = layers.Dense(self.N_CLASSES, activation='softmax', name='softmax')(x)
            self.model = Model(inputs=i.input, outputs=o, name='2d_convolution')
            self.model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        return self.model

    def LSTM(self):
        input_shape = (int(self.SR*self.DT), 1)
        i = get_melspectrogram_layer(input_shape=input_shape,
                                        n_mels=128,
                                        pad_end=True,
                                        n_fft=512,
                                        win_length=400,
                                        hop_length=160,
                                        sample_rate=self.SR,
                                        return_decibel=True,
                                        input_data_format='channels_last',
                                        output_data_format='channels_last',
                                        name='2d_convolution')
        x = LayerNormalization(axis=2, name='batch_norm')(i.output)
        x = TimeDistributed(layers.Reshape((-1,)), name='reshape')(x)
        s = TimeDistributed(layers.Dense(64, activation='tanh'),
                            name='td_dense_tanh')(x)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),
                                name='bidirectional_lstm')(s)
        x = layers.concatenate([s, x], axis=2, name='skip_connection')
        x = layers.Dense(64, activation='relu', name='dense_1_relu')(x)
        x = layers.MaxPooling1D(name='max_pool_1d')(x)
        x = layers.Dense(32, activation='relu', name='dense_2_relu')(x)
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dropout(rate=0.2, name='dropout')(x)
        x = layers.Dense(32, activation='relu',
                            activity_regularizer=l2(0.001),
                            name='dense_3_relu')(x)
        if self.N_CLASSES==2:
            o = layers.Dense(1, activation='sigmoid', name='sigmoid')(x)
            self.model = Model(inputs=i.input, outputs=o, name='long_short_term_memory')
            self.model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        else:
            o = layers.Dense(self.N_CLASSES, activation='softmax', name='softmax')(x)
            self.model = Model(inputs=i.input, outputs=o, name='long_short_term_memory')
            self.model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        
        return self.model

    def summary(self):
        print(self.model.summary())

def Summary():
    CustomModel().Conv1D().summary()
    CustomModel().Conv2D().summary()
    CustomModel().LSTM().summary()

if __name__ == "__main__":
    summary = False
    parser = argparse.ArgumentParser(description="Print the summary of the models")
    parser.add_argument("--summary", "-sum", action="store_true", help="Add argument to see summary of the models")

    args = parser.parse_args()
    summary = args.summary

    if summary:
        Summary()
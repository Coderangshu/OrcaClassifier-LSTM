import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import CustomModel as CM
from tqdm import tqdm
from glob import glob
import argparse
import warnings
import sys

parser = argparse.ArgumentParser(description='Audio Classification Training')
parser.add_argument('--model_type', type=str, default='lstm',help='model to run. i.e. conv1d, conv2d, lstm')
parser.add_argument('--src_root', type=str, default='clean',help='directory of audio files in total duration')
parser.add_argument('--batch_size', type=int, default=16,help='batch size')
parser.add_argument('--delta_time', '-dt', type=float, default=2.0,help='time in seconds to sample audio')
parser.add_argument('--sample_rate', '-sr', type=int, default=20000,help='sample rate of clean audio')
parser.add_argument('--plt_grph', '-pg', action="store_true", help="Set to plot graph of metrics")
parser.add_argument('--force', '-f', action="store_true",help="Set to train model even if present")

args, _ = parser.parse_known_args()

force = False
force = args.force
if os.path.exists("models/{}.h5".format(args.model_type)) and not force is True:
    print("Model is trained......exiting")
    sys.exit()

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical, Sequence

class DataGenerator(Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            _, wav = wavfile.read(path)
            X[i,] = wav.reshape(-1, 1)
            Y[i,] = label
            if self.n_classes>2:
                Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

def train(args):
    src_root = args.src_root
    sr = args.sample_rate
    dt = args.delta_time
    batch_size = args.batch_size
    model_type = args.model_type
    params = {'N_CLASSES':len(os.listdir(args.src_root)),
              'SR':sr,
              'DT':dt}
    models = {'conv1d':CM(**params).Conv1D(),
              'conv2d':CM(**params).Conv2D(),
              'lstm':  CM(**params).LSTM()}
    assert model_type in models.keys(), '{} is not an available model'.format(model_type)
    
    if os.path.exists('logs') is False:
        os.mkdir('logs')
    csv_path = os.path.join('logs', '{}_history.csv'.format(model_type))

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    
    classes = sorted(os.listdir(args.src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)
    
    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,labels,test_size=0.1,random_state=0)

    assert len(label_train) >= args.batch_size, 'Number of train samples must be >= batch_size'
    if len(set(label_train)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in training data. Increase data size or change random_state.'.format(len(set(label_train)), params['N_CLASSES']))
    if len(set(label_val)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in validation data. Increase data size or change random_state.'.format(len(set(label_val)), params['N_CLASSES']))

    tg = DataGenerator(wav_train, label_train, sr, dt,params['N_CLASSES'], batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt,params['N_CLASSES'], batch_size=batch_size)
    
    model = models[model_type]
    if os.path.exists('models') is False:
        os.mkdir('models')
    cp = ModelCheckpoint('models/{}.h5'.format(model_type), monitor='val_loss',save_best_only=True, save_weights_only=False,mode='auto', save_freq='epoch', verbose=1)
    csv_logger = CSVLogger(csv_path, append=False)
    model.fit(tg, validation_data=vg,epochs=30, verbose=1,callbacks=[csv_logger, cp])

def plot_history(plt_grph = False):
    if plt_grph:
        log_csvs = sorted(glob('logs/*.csv'))
        print(log_csvs)

        labels = ['Conv 1D', 'Conv 2D', 'LSTM']
        colors = ['r', 'm', 'c']

        fig, ax = plt.subplots(1, 3, sharey=True, figsize=(16,5))

        for i, (fn, label, c) in enumerate(zip(log_csvs, labels, colors)):
            # csv_path = os.path.join('..', 'logs', fn)
            csv_path = fn
            df = pd.read_csv(csv_path)
            ax[i].set_title(label, size=16)
            ax[i].plot(df.accuracy, color=c, label='train')
            ax[i].plot(df.val_accuracy, ls='--', color=c, label='test')
            ax[i].legend(loc='upper left')
            ax[i].tick_params(axis='both', which='major', labelsize=12)
            ax[i].set_ylim([0,1.0])

        fig.text(0.5, 0.02, 'Epochs', ha='center', size=14)
        fig.text(0.08, 0.5, 'Accuracy', va='center', rotation='vertical', size=14)

        if os.path.exists('logs/metric.png') is True:
            os.remove('logs/metric.png')
        plt.savefig('logs/metric.png')

if __name__ == '__main__':

    train(args)
    plot_history(args.plt_grph)
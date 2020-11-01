import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm
from scipy.io.wavfile import read
from tempfile import mktemp
import wavio
import noisereduce as nr
from librosa.core import resample, to_mono

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y[0]).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def downsample_mono(path, sr):
    obj = wavio.read(path)
    wav = obj.data.astype(np.float32, order='F')
    rate = obj.rate
    try:
        # tmp = wav.shape[1]
        wav = to_mono(wav.T)
    except:
        pass
    wav = resample(wav, rate, sr)
    wav = wav.astype(np.int16)
    return sr, wav

def noise_reduction(array):
    # wname = mktemp('.wav')
    # call.export(wname, format="wav")
    noisy_part = array
    reduced_noise = nr.reduce_noise(audio_clip=array.astype('float64'), noise_clip=noisy_part.astype('float64'), use_tensorflow=True, verbose=False)
    return reduced_noise

def make_prediction(args):

    model = load_model(args.model_fn,custom_objects={'STFT':STFT,'Magnitude':Magnitude,'ApplyFilterbank':ApplyFilterbank,'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    # labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    # le = LabelEncoder()
    # y_true = le.fit_transform(labels)
    # results = []

    for _, wav_fn in enumerate(wav_paths):
        Frequency,array = read(wav_fn)
        reduced_noise = noise_reduction(array)
        wname = mktemp('.wav')
        wavio.write(wname, reduced_noise, Frequency, sampwidth=2)
        rate, wav = downsample_mono(wname, args.sr)
        mask = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav * mask
        step = int(args.sr*args.dt)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        real_class = os.path.dirname(wav_fn).split('/')[-1]
        if args.n_class == 2:
            print('Actual class: {}, Predicted class: {}'.format(real_class, 0 if y_pred<0.5 else 1))
        else:
            y_mean = np.mean(y_pred, axis=0)
            y_pred = np.argmax(y_mean)
            print('Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))
        # results.append(y_mean)

    # np.save(os.path.join('logs', args.pred_fn), np.array(results))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/lstm.h5',help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred',help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='clean',help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=2.0,help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=20000,help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,help='threshold magnitude for np.int16 dtype')
    parser.add_argument('--n_class','-nc',type=int,default=2,help='number of classes to be classified')

    args, _ = parser.parse_known_args()

    make_prediction(args)
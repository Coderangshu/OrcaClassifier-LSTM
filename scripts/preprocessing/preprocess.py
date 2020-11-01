#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!python
import matplotlib.pyplot as plt
from glob import glob
import argparse
import os
import sys
from pathlib import Path
import selection_table as sl
import soundfile as sf
import librosa
from librosa.core import resample, to_mono
import pandas as pd
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
from tempfile import mktemp
from scipy.io import wavfile
from scipy.io.wavfile import read
import wavio
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser(description="Preprocess audio files for use with CNN models")
parser.add_argument("--tsv_path",type=str,help="Path to tsv file")
parser.add_argument("--files_dir",type=str,help="Path to directory with audio files")
parser.add_argument("--call_time",type=int, default=2, help="Target length of processed audio file")
parser.add_argument("--output_dir",type=str, default='extracted-calls', help="Path to output directory")
parser.add_argument("--reduce_noise","-nr",action="store_true",help="Set true: Reduce noise in extracted calls")

# parser.add_argument('--src_root', type=str, default='../extracted-calls',help='directory of audio files in total duration')
parser.add_argument('--dst_root', type=str, default='clean',help='directory to put audio files split by delta_time')
parser.add_argument('--delta_time', '-dt', type=float, default=2.0,help='time in seconds to sample audio')
parser.add_argument('--sr', type=int, default=20000,help='rate to downsample audio')
# parser.add_argument('--fn', type=str, default='../extract_calls/extracted_calls0.wav',help='file to plot over time to check magnitude')
parser.add_argument('--threshold', type=str, default=20,help='threshold magnitude for np.int16 dtype')

args = parser.parse_args()

if os.path.exists(args.dst_root) is True:
    print("Dataset is up-to-date.....exiting")
    sys.exit()

# In[ ]:
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

def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    # dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    dst_path = os.path.join(target_dir, fn+'_{}.wav'.format(str(ix)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)

def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

def split_wavs(args):
    src_root = args.output_dir
    dst_root = args.dst_root
    dt = args.delta_time

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    # dirs = os.listdir(src_root)
    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir),desc="Further cleaning the extracts: "):
            src_fn = os.path.join(src_dir, fn)
            rate, wav = downsample_mono(src_fn, args.sr)
            mask = envelope(wav, rate, threshold=args.threshold)
            wav = wav * mask
            delta_sample = int(dt*rate)

            # cleaned audio is less than a single sample
            # pad with zeros to delta_sample size
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,1), dtype=np.int16)
                # print(sample.shape)
                sample[:wav.shape[0]] = wav[0]
                save_sample(sample, rate, target_dir, fn, 0)
            # step through audio and save every delta_sample
            # discard the ending audio if it is too short
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav[start:stop]
                    save_sample(sample, rate, target_dir, fn, cnt)

# def test_threshold(args):
#     src_root = args.src_root
#     wav_paths = glob('{}/**'.format(src_root), recursive=True)
#     wav_path = [x for x in wav_paths if args.fn in x]
#     if len(wav_path) != 1:
#         print('audio file not found for sub-string: {}'.format(args.fn))
#         return
#     rate, wav = downsample_mono(wav_path[0], args.sr)
#     mask = envelope(wav, rate, threshold=args.threshold)
#     plt.style.use('ggplot')
#     plt.title('Signal Envelope, Threshold = {}'.format(str(args.threshold)))
#     plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
#     plt.plot(wav[mask], color='c', label='keep')
#     plt.plot(env, color='m', label='envelope')
#     plt.grid(False)
#     plt.legend(loc='best')
#     plt.show()

def generate_negative_tsv(call_annotations,call_time,files_dir):

    """Generates .tsv file containing start-time and end-time of
    the negative calls.
    Since we also want the pure negative samples, that do not contain
    the calls we would generate a .tsv file which contains the interval
    not in the start-time and duration of the .tsv containing the calls.
    And since any area that does not contain the calls would contain no
    call or the background noise, we would use this start-time and
    duration to extract audio from the audio files.
    Args:
        call_annotations: The .tsv file containing the calls.
        call_time: The duration for which you want to generate negative
                    calls.
        files_dir: The directory that contains the audio data.
    Returns:
        A pandas dataframe containing start_time and end_time of the
        background sounds.
    """
    standardized_annotations = sl.standardize(table=call_annotations,signal_labels=["SRKWs"],mapper={"wav_filename": "filename"},trim_table=True)

    positives_call_duration = sl.select(annotations=standardized_annotations,length=call_time)
    file_durations = sl.file_duration_table(files_dir)

    # Generate a .tsv file which does not include any calls.
    negatives_annotations = sl.create_rndm_backgr_selections(annotations=standardized_annotations,files=file_durations,length=call_time,num=len(positives_call_duration),trim_table=True)

    negative_tsv_generated = negatives_annotations.reset_index(level=[0, 1])

    return negative_tsv_generated


# In[ ]:
def noise_reduction(output_file,call):
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    wname = mktemp('.wav')
    call.export(wname, format="wav")
    (Frequency, array) = read(wname)
    noisy_part = array
    reduced_noise = nr.reduce_noise(audio_clip=array.astype('float64'), noise_clip=noisy_part.astype('float64'), use_tensorflow=True, verbose=False)

    wavio.write(output_file, reduced_noise, Frequency, sampwidth=2)

def extract_audio(output_directory,file_location,call_time_in_seconds,call_annotations,reduce_noise=False):
    """This function extracts the audio of a specified duration.
    Since a single audio clip might consist of a mixture of both calls
    and no calls, therefore smaller audio clips of particular time frame
    are extracted to get the complete positive and negative calls. These
    calls are extracted by taking the start-time from the .tsv file and
    the duration of the call as specified by the user.
    Args:
        label: A string specifying whether we are extracting calls or
            no call.
        tsv_filename: The .tsv file containing the parameters like start-time,
            duration, etc.
        output_directory: The path output directory where we want to store
            these extracted calls.
        file_location: The location of the audio file in .wav format.
        call_time_in_seconds: Enter the duration of calls you want
            to extract in seconds.Integer value.
    Returns:
        None
    """
    if output_directory[-1]=='1':
        dir = "Positive calls"
    else:
        dir = "Negative calls"

    file_name = call_annotations.filename[:].values
    start_time = call_annotations.start[:].values

    i = 0
    call_duration = 0
    call_time_in_seconds = call_time_in_seconds*1000

    for i in tqdm(range(len(file_name)),desc = "{} extraction".format(dir)):

        output_file = os.path.join(output_directory,"extracted_calls{0}.wav".format(i))
        if os.path.exists(output_file):
            continue

        audio_file = file_name[i]
        audio_file = os.path.join(file_location, audio_file)
        sound = AudioSegment.from_file(audio_file)
        start_time_duration = start_time[i]
        start_time_duration = start_time_duration * 1000
        i = i + 1
        call_duration = start_time_duration + call_time_in_seconds
        call = sound[start_time_duration:call_duration]

        if reduce_noise:
            noise_reduction(output_file,call)
        else:
            call.export(output_file, format="wav")

# In[ ]:


def main(tsv_path,files_dir,call_time,output_dir,reduce_noise):

    # prepare output directories
    positive_dir = os.path.join(output_dir, "1")
    if os.path.exists(positive_dir) is False:
        os.mkdir(positive_dir)

    negative_dir = os.path.join(output_dir, "0")
    if os.path.exists(negative_dir) is False:
        os.mkdir(negative_dir)

    # load tsv file
    call_annotations = pd.read_csv(tsv_path, sep="\t")
    call_annotations.rename(columns={'start_time_s':'start'}, inplace=True)
    call_annotations.rename(columns={'wav_filename':'filename'}, inplace=True)
    call_annotations["label"] = "SRKWs"
    try:
        call_length_mean = call_annotations["duration_s"].mean()
        print("The mean of the call duration is {}".format(call_length_mean))
    except Exception:
        print("Please change the call duration label in your .tsv file by 'duration_s' ")
    try:
        call_annotations["end"] = call_annotations["start"] + call_annotations["duration_s"]
    except Exception:
        print("Please change the start time of the call label in your .tsv to start")

    # extract the audio of the calls
    extract_audio(positive_dir,files_dir,call_time,call_annotations,reduce_noise)

    # generate negative .tsv file
    negative_generated_tsv = generate_negative_tsv(call_annotations,call_time, files_dir)

    # extract the audio of the negative calls or background calls

    extract_audio(negative_dir,files_dir,call_time,negative_generated_tsv,reduce_noise)

    # test_threshold(args)
    split_wavs(args)

# In[ ]:


if __name__ == "__main__":

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)
    
    main(args.tsv_path,args.files_dir,args.call_time,args.output_dir,args.reduce_noise)
    
    shutil.rmtree(args.output_dir)

# In[ ]:

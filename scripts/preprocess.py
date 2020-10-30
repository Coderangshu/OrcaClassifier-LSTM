#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!python
import argparse
import os
from pathlib import Path
import selection_table as sl
import soundfile as sf
import librosa
import pandas as pd
from pydub import AudioSegment
import noisereduce as nr
from tempfile import mktemp
from scipy.io.wavfile import read
import wavio
from tqdm import tqdm


# In[ ]:


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
        audio_file = file_name[i]
        audio_file = os.path.join(file_location, audio_file)
        sound = AudioSegment.from_file(audio_file)
        start_time_duration = start_time[i]
        start_time_duration = start_time_duration * 1000
        i = i + 1
        call_duration = start_time_duration + call_time_in_seconds
        call = sound[start_time_duration:call_duration]

        if reduce_noise:
            import tensorflow as tf
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            wname = mktemp('.wav')
            call.export(wname, format="wav")
            (Frequency, array) = read(wname)
            noisy_part = array
            reduced_noise = nr.reduce_noise(audio_clip=array.astype('float64'), noise_clip=noisy_part.astype('float64'), use_tensorflow=True, verbose=False)

            output_file = os.path.join(
                            output_directory,
                            "extracted_calls{0}.wav".format(i))
            wavio.write(output_file, reduced_noise, Frequency, sampwidth=2)

        else:
            output_file = os.path.join(
                            output_directory,
                            "extracted_calls{0}.wav".format(i))
            call.export(output_file, format="wav")

# In[ ]:


def main(tsv_path,files_dir,call_time,output_dir,reduce_noise):

    # prepare output directories
    positive_dir = os.path.join(output_dir, "1")
    if not os.path.isdir(positive_dir):
        os.mkdir(positive_dir)

    negative_dir = os.path.join(output_dir, "0")
    if not os.path.isdir(negative_dir):
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


# In[ ]:


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess audio files for use with CNN models")
    parser.add_argument("--tsv_path",type=str,help="Path to tsv file")
    parser.add_argument("--files_dir",type=str,help="Path to directory with audio files")
    parser.add_argument("--call_time",type=int,help="Target length of processed audio file")
    parser.add_argument("--output_dir",type=str,help="Path to output directory")
    parser.add_argument("--reduce_noise",action="store_true",help="Set true: Reduce noise in extracted calls")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main(args.tsv_path,args.files_dir,args.call_time,args.output_dir,args.reduce_noise)


# In[ ]:

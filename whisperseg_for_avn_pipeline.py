# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:09:43 2024

@author: maria
"""
#change this file path based on where WhisperSeg was cloned on your system. 
import sys
sys.path.insert(0, r"PATH/TO/WSEG/ON/LOCAL/MACHINE") 

import shutil
import librosa
import json
import glob
import pandas as pd
from audio_utils import SpecViewer
import os
import torch

spec_viewer = SpecViewer()
from model import WhisperSegmenterFast
segmenter = WhisperSegmenterFast( "nccratliri/whisperseg-large-ms-ct2", device="cuda" )


sr = 32000 
min_frequency = 0
spec_time_step = 0.0025
min_segment_length = 0.01
eps = 0.02
num_trials = 3




def wseg_one_round(song_folder):

    bird_list = os.listdir(song_folder)
    
    for bird_id in bird_list[:]:
        
        print(bird_id)
        
        song_folder_path = os.path.join(song_folder, bird_id)+"\\"
        all_songs = glob.glob(song_folder_path + "*.wav")
        
        full_seg_table = pd.DataFrame()
        
        for i, song in enumerate(all_songs):
            #load audio
            audio, __ = librosa.load(song, sr = sr)

            #segment file
            prediction = segmenter.segment(audio, 
                                          sr = sr,
                                          min_frequency = min_frequency, 
                                          spec_time_step = spec_time_step,
                                          min_segment_length = min_segment_length, 
                                          eps = eps, 
                                          num_trials = num_trials)

            #format segmentation as dataframe
            curr_prediction_df = pd.DataFrame(prediction)
            #spec_viewer.visualize( audio = audio, sr = sr, min_frequency= min_frequency, prediction = prediction,window_size=10, precision_bits=1)


            #add file name to dataframe
            song_name = song.split("\\")[-1]
            curr_prediction_df['files'] = song_name

            #add current file's segments to full_seg_table
            full_seg_table = pd.concat([full_seg_table, curr_prediction_df])
            print(song,"\nPrediction:\n", prediction, "\n")
            
        column_names = ['onsets', 'offsets', 'cluster', 'files']
        full_seg_table.columns = column_names
        seg_folder = os.path.join(song_folder_path, "Segmentations")
        if os.path.exists(seg_folder):
            shutil.rmtree(seg_folder)
        os.makedirs(seg_folder)
        csv_name = bird_id + ".csv"
        csv_file_path = os.path.join(seg_folder, csv_name)             
        full_seg_table.to_csv(csv_file_path) #set path to output file according to your file system. 
        
if __name__ == "__main__":

      #parameters to change

      #Parent directory with all rounds (for sim between rounds) or bird to self-compare
      #Parent directory should contain rounds as subdirectories 
      directory_path = r'PATH/TO/DIRECTORY/WITH/ROUND'
     
      for subdirectory in os.listdir(directory_path):
          if subdirectory[0] != '.':
              print("______"+subdirectory+"_______")
              print("Calling wseg_one_round on", subdirectory)
              subdirectory_path = os.path.join(directory_path, subdirectory)  
              wseg_one_round(subdirectory_path)

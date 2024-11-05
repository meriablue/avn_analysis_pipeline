# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:37:40 2024

@author: Maria Pescaru, in Sakata-Woolley Lab 
"""

import avn.similarity as similarity
import pandas as pd
import os
import csv
from itertools import combinations

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random 


def get_seg_file(song_folder):
    '''
    Function takes as input a song_directory and returns path to segmentation file
    :param song_folder: path to directory with .wav files and segmentation
    :return: string
    '''
    parent_directory, directory_name = os.path.split(song_folder)
    segmentations_dir = os.path.join(song_folder, 'Segmentations')
    seg_file_path = os.path.join(segmentations_dir, f"{directory_name}.csv")
    print(song_folder+'\n' + seg_file_path+'\n'+directory_name+'\n')
    return seg_file_path, directory_name


def spect_prep(song_folder):
    '''
    Function takes as input the path to a song folder and splits the original 
    Segmentation file into Segmentation files for each song for further processing
    '''
    seg_file_path, bird_id = get_seg_file(song_folder)
    segmentations = pd.read_csv(seg_file_path)
    segmentations.head()
    out_dir = os.path.join(song_folder, 'Similarity_Prep')
    grouped = segmentations.groupby('files')
    print(grouped.groups.keys())
    
    for file in os.listdir(song_folder):
        if file.endswith('.wav') and file in grouped.groups.keys():                                                           
            seg_file_one_song = grouped.get_group(file)
            out_dir_one_file = os.path.join(out_dir, file)
            song_folder = song_folder+'\\'
            similarity.prep_spects(Bird_ID=bird_id, segmentations=seg_file_one_song, song_folder_path=song_folder,
                               out_dir=out_dir_one_file)
    return out_dir
        

def similarity_score(song_folder, output_dir):
    '''
    Main function of program, creates a csv file in the output directory 
    containing 1000 EMD scores for random subsets of the spectrograms
    :param song_folder: folder with .wav files and segmentation
    :return:string: path to similarity prep folder
    '''
    #Create csv file
    parent_directory, dir_name = os.path.split(song_folder)
    csv_file_path = os.path.join(output_dir, f"{dir_name}.csv")
    
    #calculate embeddings
    model=similarity.load_model()
    spect_dir=spect_prep(song_folder)
    embeddings=[]
    i = 1; 
    all_syllables = 0; 
    
    for subdir in os.listdir(spect_dir):
        spect_dir_one_file = os.path.join(spect_dir, subdir)
        if os.path.isdir(spect_dir_one_file):
            print("___________"+spect_dir_one_file+"____________")
            embeddings_one_file = similarity.calc_embeddings(Bird_ID = dir_name, 
                                                         spectrograms_dir = spect_dir_one_file, 
                                                         model=model)
            embeddings.append(embeddings_one_file)
            all_syllables = all_syllables + embeddings_one_file.shape[0]
            print("EMBEDDINGS SHAPE:", embeddings_one_file.shape)
            #embeddings = similarity.calc_embeddings(Bird_ID = dir_name, 
                                            #spectrograms_dir=spect_dir,
                                            #model=model)
            i=i+1
    
    nb_files = i - 1
    
    index_size = int(np.floor(nb_files/2))
    
        
    
    
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([song_folder])
        csv_writer.writerow(['', 'EMD'])
        indices = list(range(0, nb_files))
       
        #generate splits
        for i in range(1000):
            
            
            #print("INDEX ARRAY INDEX ARRAY INDEX ARRAY")

            #print(indices)
            random.shuffle(indices)
            
            split1_indices = np.random.choice(indices, index_size, replace=False)
            split2_indices = np.setdiff1d(indices, split1_indices)
            
            split_size = int(np.floor(all_syllables/2))
            split1 = np.empty((split_size, 8))
            split2 = np.empty((split_size, 8))
            
            
            counter_split1 = 0
            counter_split2 = 0
            
            #print("INDEX ARRAYS BEING COMPARED\n\n\n")
            #print(split1_indices)
            #print(split2_indices)
            
            for j in range(nb_files-1):
                embeddings_one_file = np.copy(embeddings[j])
                (nb_syll_one_file, nb_emb) = embeddings_one_file.shape
                if j in split1_indices:
                    
                    for k in range (nb_syll_one_file - 1):
                        if counter_split1 < split_size : 
                            #print (embeddings_one_file)
                            #print("NOW SUPOPSEDLY ONE ROW")
                            #print (embeddings_one_file[k])
                            split1[counter_split1] = np.copy(embeddings_one_file[k])
                            counter_split1 = counter_split1 + 1 
                    
                else:
                    
                    for k in range (nb_syll_one_file-1):
                        if counter_split2 < split_size : 
                            #print (embeddings_one_file)
                            #print("NOW SUPOPSEDLY ONE ROW")
                            #print (embeddings_one_file[k])
                            split2[counter_split2] = np.copy(embeddings_one_file[k])
                            counter_split2 = counter_split2 + 1
           
            emd = similarity.calc_emd(split1, split2)
            
            print("EMD SCORE:", emd)
            
            csv_writer.writerow([i, emd])
            
            
if __name__ == "__main__":

            
#Parameters to change:

    song_directory = r'DIRECTORY\WITH\SONG\FILES'
    output_directory = r'DIRECTORY\FOR\OUTPUT'
    
   #for subdir in os.listdir(song_directory):
        
    #   subdir_path = os.path.join(song_directory, subdir)
        
    for subdir in os.listdir(song_directory):
                
                sub_path = os.path.join(song_directory, subdir)
                similarity_score(sub_path, output_directory)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

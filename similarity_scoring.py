import avn.similarity as similarity
import pandas as pd
import os
import csv
from itertools import combinations

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


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


def calculate_embeddings(song_folder, embeddings_arr):
    '''
    Function stores embeddings and stores them in embeddings_arr
    :param song_folder: folder with .wav files and segmentation
    :param embeddings_arr: array of embeddings that will be populated
    :return:
    '''
    # Get the segmentation files and bird ID for each of the birds
    seg_file_path, bird_id = get_seg_file(song_folder)
    
    

    segmentations = pd.read_csv(seg_file_path)
    segmentations.head()
    out_dir = os.path.join(song_folder, 'Similarity_Prep')
    song_folder = song_folder+'\\'
    similarity.prep_spects(Bird_ID=bird_id, segmentations=segmentations, song_folder_path=song_folder,
                           out_dir=out_dir)
    model = similarity.load_model()
    embeddings_arr = similarity.calc_embeddings(Bird_ID=bird_id, spectrograms_dir=out_dir, model=model)


def similarity_scoring(song_folder_1, song_folder_2):
    '''
    Function computes similarity scores for 2 birds and returns emd score
    :param song_folder_1: path to .wav and segmentation folder of bird 1
    :param song_folder_2: path to .wav and segmentation folder of bird 2
    :return: float
    '''

    #Get embeddings for each bird
    embeddings_1 = np.array([[]])
    embeddings_2 = np.array([[]])
   
    
    #Calculate embeddings song 1
    seg_file_path, bird_id = get_seg_file(song_folder_1)

    segmentations = pd.read_csv(seg_file_path)
    segmentations.head()
    out_dir = os.path.join(song_folder_1, 'Similarity_Prep')
    song_folder = song_folder_1+'\\'
    similarity.prep_spects(Bird_ID=bird_id, segmentations=segmentations, song_folder_path=song_folder,
                           out_dir=out_dir)
    model = similarity.load_model()
    embeddings_1 = similarity.calc_embeddings(Bird_ID=bird_id, spectrograms_dir=out_dir, model=model)
    
    
    #Caluclate embeddinghs song 2 
    seg_file_path, bird_id = get_seg_file(song_folder_2)

    segmentations = pd.read_csv(seg_file_path)
    segmentations.head()
    out_dir = os.path.join(song_folder_2, 'Similarity_Prep')
    song_folder = song_folder_2+'\\'
    similarity.prep_spects(Bird_ID=bird_id, segmentations=segmentations, song_folder_path=song_folder,
                           out_dir=out_dir)
    model = similarity.load_model()
    embeddings_2 = similarity.calc_embeddings(Bird_ID=bird_id, spectrograms_dir=out_dir, model=model)
    

    #Calculate emd score
    emd = similarity.calc_emd(embeddings_1, embeddings_2)
    print("________")
    print(embeddings_1)
    return emd


def similarity_scoring_directory(dir_path, output_dir_path):
    '''
    Function creates a .csv file in the output directory with comparisons
    between all birds in given directory
    :param bird_dir_path: Path to directory containing subdirectories with .wav files
    :param output_dir_path: Path to directory where output .csv file will be saved
    :return: noneo
    '''

    #Create csv file
    parent_directory, dir_name = os.path.split(dir_path)
    csv_file_path = os.path.join(output_dir_path, f"{dir_name}.csv")

    #get all subdirectories
    subdirectories = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if
                      os.path.isdir(os.path.join(dir_path, d))]
    print(subdirectories)
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Bird1', 'Bird2', 'SimilarityScore'])

        for dir1, dir2 in combinations(subdirectories, 2):
            print("Calling similarity_scoring on", dir1, dir2)
            score = similarity_scoring(dir1, dir2)
            subdir1_name = os.path.basename(dir1)
            subdir2_name = os.path.basename(dir2)
            csv_writer.writerow([subdir1_name, subdir2_name, score])

    print(f"CSV file created at: {csv_file_path}")


if __name__ == "__main__":

    
    #Path to directory containing songs for birds you want to compare
    directory_path = r'PATH/TO/PARENT/DIRECTORY'

    #Directory where comparison for each folder will be saved in a separate csv file
    output_path = r'PATH/TO/OUTPUT'

    #Boolean variable, should be set to 0 for a single round, otherwise set to one, see README for more details 
    multiple_rounds = 
    
    if ! multiple_directories : 
        
        for subdirectory in os.listdir(directory_path):
                similarity_scoring_directory(directory_path, output_path)
        
    else:
    #Loop through subdirectories and call similarity_scoring
        #similarity_scoring_directory(directory_path, output_path)

        for subdirectory in os.listdir(directory_path):
            if subdirectory[0] != '.':
                print("______"+subdirectory+"_______")
                print("Calling similarity_scoring_directory on", subdirectory)
                subdirectory_path = os.path.join(directory_path, subdirectory)  
                similarity_scoring_directory(subdirectory_path, output_path)



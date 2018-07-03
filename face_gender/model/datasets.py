from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2

class Center_control(object):
    
    def __init__(self, file_name='imdb', file_path=None, image_size=(48, 48)):

        self.file_name = file_name
        self.file_path = file_path
        self.image_size = image_size
        if self.file_path != None:
            self.file_path = file_path
        elif self.file_name == 'imdb':
            self.file_path = 'C:/Users/Saif/Desktop\python/machine learnig/Transfer-Learning-in-keras---custom-data-master/face_classification-master/face_classification-master/datasets/imdb_crop/imdb_crop/imdb.mat'
        

    def obtainment(self):
        if self.file_name == 'imdb':
            Database = self.loading_imdb()
        return Database 
        

    def loading_imdb(self):
        face_score_treshold = 3
        dataset = loadmat(self.file_path)
        image_names_array = dataset['imdb']['full_path'][0, 0][0]
        gender_classes = dataset['imdb']['gender'][0, 0][0]
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
        return dict(zip(image_names, gender_classes))

    
def get_gender(file_name):
    
    if file_name == 'imdb':
        return {0:'WOMEN', 1:'man'}
   

def divide_imdb(Database, validation_split=.2, do_shuffle=False):
    base_keys = sorted(Database.keys())
    if do_shuffle == True:
        shuffle(base_keys)
    training_split = 1 - validation_split
    num_train = int(training_split * len(base_keys))
    train_data = base_keys[:num_train]
    validation_data = base_keys[num_train:]
    return train_data, validation_data

    


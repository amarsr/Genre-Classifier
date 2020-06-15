#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import IPython.display as ipd
import audioread
import csv
import librosa
from tqdm import tqdm_notebook
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

home = "C:\\Users\\Amar Srinivasan\\Documents\\Music-Analysis-Project"


# In[2]:


genre_dict = {}


# In[3]:


os.chdir(home)
with open("genre_map.csv","r") as genre_map:
    reader = csv.reader(genre_map)
    for row in reader:
        genre_dict[row[0]] = row[1]


# ### Constructing the Dataset

# In[4]:


headers = ["Audio Mean", "Zero Crossings", "Spectral Centroid", "Rolloff"]
for i in range(1,21):
    headers.append(f"MFCC {i}")
headers.append("Genre")


# In[15]:


os.chdir(home + "\\fma_small")
folders = os.listdir()
with open("song_data.csv", 'w', newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for f in folders:
        if f not in ['checksums', 'README']:
            os.chdir(f)
            songs = os.listdir()
            print(f)
            for song in tqdm_notebook(songs):
                try:
                    song_id = song.replace(".mp3", "").lstrip('0')
                    if song_id in genre_dict.keys():
                        song_genre = genre_dict[song_id]

                        ### FEATURE SELECTION PORTION ###
                        audio,sr = librosa.load(song)
                        audio = np.trim_zeros(audio)
                        audio_mean = audio.mean()
                        zero_crossings = sum(librosa.zero_crossings(audio, pad = False))
                        spectral_centroid = librosa.feature.spectral_centroid(audio, sr = sr).mean()              
                        mfcc = librosa.feature.mfcc(audio, sr = sr)
                        rolloff = librosa.feature.spectral_rolloff(audio, sr = sr).mean()
                        mfcc_list = [np.mean(e) for e in mfcc]

                        data_list = [audio_mean, zero_crossings, spectral_centroid, rolloff]
                        data_list += mfcc_list + [song_genre]
                        writer.writerow(data_list)
                except:
                    print(f"Error with song {song_id} in folder {f}.")
            os.chdir("..")


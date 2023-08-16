from librosa.core import audio
from scipy.signal import waveforms
from torch.utils.data.dataset import Dataset
import librosa
from glob import glob
import cv2
import numpy as np
import torch 
import random
import pandas as pd
import time
import clip

from textaugment import EDA
import nltk
import pickle
from PIL import Image
import os

class AudioSetTestDataset(Dataset) :
    def __init__(
        self,
        seed = 1,
        dataset_path = "./audioset",
        prompt_generator = lambda x : x
    ) :
        super(AudioSetTestDataset, self).__init__()
        self.dataset_path = dataset_path
        self.audio_path_list = sorted(glob(os.path.join(self.dataset_path, "*.wav")))
        self.label_list = list(map(
            # format of each audio file is "[label]_[index of that audio file].wav" 
            lambda file_path : os.path.basename(file_path).split('_')[0],
            self.audio_path_list
        ))

        # fix random
        random.seed(seed)

        self.prompt_generator = prompt_generator
        self.time_length = 864
        self.n_mels = 128
        self.text_aug = EDA()
        self.width_resolution = 512


    def __getitem__(self, idx):
        audio_inputs, sr = librosa.load(self.audio_path_list[idx])
        audio_inputs = librosa.feature.melspectrogram(y=audio_inputs, sr=sr, n_mels=self.n_mels)
        audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1
        audio_inputs = np.array([audio_inputs])


        text_prompt = self.label_list[idx]

        c, h, w = audio_inputs.shape

        if w >= self.time_length:
            j = random.randint(0, w-self.time_length)
            audio_inputs = audio_inputs[:,:,j:j+self.time_length]
        elif w < self.time_length:
            zero = np.zeros((1, self.n_mels, self.time_length))
            j = random.randint(0, self.time_length - w - 1)
            zero[:,:,j:j+w] = audio_inputs[:,:,:w]
            audio_inputs = zero
       
        audio_inputs = cv2.resize(audio_inputs[0], (self.n_mels, self.width_resolution))
        audio_inputs = cv2.resize(audio_inputs[0], (self.n_mels, self.width_resolution))
        
        # disabled augment process for test dataset
        #audio_aug = self.spec_augment(audio_inputs)
        audio_inputs = audio_inputs.reshape(-1, self.n_mels, self.width_resolution)
        #audio_aug = audio_aug.reshape(-1, self.n_mels, self.width_resolution)
            
        audio_inputs = torch.from_numpy(audio_inputs).float()
        #audio_aug = torch.from_numpy(audio_aug).float()

        # disable text augment process for test dataset
        #text_prompt = self.text_aug.synonym_replacement(text_prompt)
        #text_prompt = self.text_aug.random_swap(text_prompt)
        #text_prompt = self.text_aug.random_insertion(text_prompt)


        # user can edit text prompt by pass function prompt_generator()
        text_prompt = self.prompt_generator(text_prompt)

        #return audio_inputs, audio_aug, text_prompt
        return audio_inputs, text_prompt

    def spec_augment(self, spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0
        return spec

    def __len__(self):
        return len(self.audio_path_list)




class VGGSoundTestDataset(Dataset) :
    def __init__(
        self,
        seed = 1,
        dataset_path = "./vggsound",
        prompt_generator = lambda x : x
    ) :
        super(VggSoundTestDataset, self).__init__()
        self.dataset_path = dataset_path
        self.audio_path_list = sorted(glob(os.path.join(self.dataset_path, "*.wav")))
        self.label_list = list(map(
            # format of each audio file is "[label]_[index of that audio file].wav" 
            lambda file_path : os.path.basename(file_path).split('_')[0],
            self.audio_path_list
        ))

        # fix random
        random.seed(seed)

        self.prompt_generator = prompt_generator
        self.time_length = 864
        self.n_mels = 128
        self.text_aug = EDA()
        self.width_resolution = 512 

    def __getitem__(self, idx) :
        audio_inputs, sr = librosa.load(self.audio_path_list[idx])
        audio_inputs = librosa.feature.melspectrogram(y=audio_inputs, sr=sr, n_mels=self.n_mels)
        audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1
        audio_inputs = np.array([audio_inputs])


        text_prompt = self.label_list[idx]

        c, h, w = audio_inputs.shape

        if w >= self.time_length:
            j = random.randint(0, w-self.time_length)
            audio_inputs = audio_inputs[:,:,j:j+self.time_length]
        elif w < self.time_length:
            zero = np.zeros((1, self.n_mels, self.time_length))
            j = random.randint(0, self.time_length - w - 1)
            zero[:,:,j:j+w] = audio_inputs[:,:,:w]
            audio_inputs = zero
       
        audio_inputs = cv2.resize(audio_inputs[0], (self.n_mels, self.width_resolution))
        audio_inputs = cv2.resize(audio_inputs[0], (self.n_mels, self.width_resolution))
        
        # disabled augment process for test dataset
        #audio_aug = self.spec_augment(audio_inputs)
        audio_inputs = audio_inputs.reshape(-1, self.n_mels, self.width_resolution)
        #audio_aug = audio_aug.reshape(-1, self.n_mels, self.width_resolution)
            
        audio_inputs = torch.from_numpy(audio_inputs).float()
        #audio_aug = torch.from_numpy(audio_aug).float()

        # disable text augment process for test dataset
        #text_prompt = self.text_aug.synonym_replacement(text_prompt)
        #text_prompt = self.text_aug.random_swap(text_prompt)
        #text_prompt = self.text_aug.random_insertion(text_prompt)

        # user can edit text prompt by pass function prompt_generator()
        text_prompt = self.prompt_generator(text_prompt)

        #return audio_inputs, audio_aug, text_prompt
        return audio_inputs, text_prompt


    def spec_augment(self, spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0
        return spec

    def __len__(self):
        return len(self.audio_path_list)
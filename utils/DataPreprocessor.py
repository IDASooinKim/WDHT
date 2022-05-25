#-- coding: utf-8 _*_

r"""
    @Copyright 2022
    The Intelligent Data Analysis, in DKU. All Rights Reserved.
"""

import json
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from glob import glob
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor():

    def __init__(self, img_path, label_path):
        
        self.img_path = img_path
        self.label_list = glob(label_path+'/*.json')
        
        self.img_file_name = []
        self.scene_attrib = []
        self.all_label_list = set()
        self.encoded_label = []
        self.word_vector = []

    def load_label(self):
        print("[INFO] : Loading label has been started!")
        for json_path in tqdm(self.label_list):
            
            with open(json_path, 'r') as file:
                
                label_dict = json.load(file)
                self.img_file_name.append(label_dict['annotation']['filename'])
                all_labels = label_dict['annotation']['sceneattribute'] + [label_dict['annotation']['sceneenvironment']] + list(label_dict['annotation']['scenecategory'])
                self.scene_attrib.append(all_labels)
                for single_label in all_labels:
                    self.all_label_list.add(single_label)
        print("[INFO] : Loading label has been finished!")

    def encode_label(self):
        
        set_label_list = list(self.all_label_list)
        encode_label_model = LabelEncoder()
        encode_label_model.fit(set_label_list)

        for single_img_label in self.scene_attrib:
            encoded_label = encode_label_model.transform(single_img_label)
            self.encoded_label.append(encoded_label)
    
    def word2vec_encode(self, model_path):
        word2vec_model = Word2Vec.load(model_path)
        print('[INFO] : Word2vec has been started!') 
        for word in tqdm(self.scene_attrib):
            temp = []
            for single_word in word:
                temp_single_word = word2vec_model.wv[single_word]
                temp.append(temp_single_word)
            self.word_vector.append(temp)
        print('[INFO] : Word2vec has been finished!')

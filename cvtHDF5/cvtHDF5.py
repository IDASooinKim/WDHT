import h5py
import numpy as np

from tqdm import tqdm
from torchvision.io import read_image
from torchvision.transforms.functional import resize

class cvtHDF5():

    def __init__(self, img_name, img_label, img_size):
        
        # Initialize parameters
        self.img_name = img_name
        self.img_label = img_label
        self.img_size = img_size
        self.img_list = []
        self.label_list = []

    def load_img(self):

        # load image
        for i in tqdm(range(self.img_label.shape[0]-1)):
            
            #try:
            image = resize(read_image('/data/tag_test/tag_sample_data/sample_data/' + self.img_name[i]), self.img_size)/255.
            label = self.img_label[i]
            
            #except:
            #    print("Corrupted Image : {}".format(self.img_name[i]))
            self.img_list.append(image)
            self.label_list.append(label)
        
        self.img_list = np.array(self.img_list).dtype('float64')
        self.label_list = np.array(self.label_list).dtype('float64')

        print("Total Image Data Shape : {}".format(self.img_list.shape))
        print("Total Label Data Shape : {}".format(self.label_list.shape))

    def cvt_img2hdf5(self,save_path, test_ratio, shuffle=True):

        if shuffle:
            
            idx = np.arange(self.img_list.shape[0])
            np.random.shuffle(idx)

            self.img_list = self.img_list[idx]
            self.label_list = self.label_list[idx]
        
        train_data_num = int((1-test_ratio)*self.img_list.shape[0])
        
        img_train = self.img_list[:train_data_num]
        img_test = self.img_list[train_data_num:]
        label_train = self.label_list[:train_data_num]
        label_test = self.label_list[train_data_num:]

        # Write hdf5 dataset
        hdf5 = h5py.File(save_path+'.hdf5', 'w')

        #hdf5.create_group("train")
        #hdf5.create_group("test")
        
        hdf5.create_dataset("train/images", data=img_train)
        hdf5.create_dataset("train/labels", data=label_train)
        hdf5.create_dataset("test/images", data=img_test)
        hdf5.create_dataset("test/labels", data=label_test)

        hdf5.close()

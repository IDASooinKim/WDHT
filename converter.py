import numpy as np

from cvtHDF5.cvtHDF5 import cvtHDF5

from utils.ArgParse import get_args
from utils.DataPreprocessor import DataPreprocessor
from utils.DataGenerator import ImageDataset

if __name__ =='__main__':

    parser = get_args()

    label_processor = DataPreprocessor(parser.img_path, parser.label_path)
    label_processor.load_label()
    all_label_list = label_processor.all_label_list
    img_path = label_processor.img_file_name
    img_path.remove('11034411_이인희04_국립중앙박물관_불상_00061.jpg')
    label_processor.word2vec_encode(parser.w2v_path)
    tag_word_vec = np.array(label_processor.word_vector)
    
    print("loading start")
    converter = cvtHDF5(img_path, tag_word_vec, (244,244))
    print("loading img start")
    converter.load_img()
    print("convert to hdf5 start")
    converter.cvt_img2hdf5('hdf5/tag_test', 0.2)


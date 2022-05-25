from gensim.models import Word2Vec, KeyedVectors
from utils.DataPreprocessor import DataPreprocessor


img_path = '/data/tag_tes'
label_path = '/data/tag_test/tag_sample_annotation'

if __name__ == '__main__':
    label_processor = DataPreprocessor(img_path, label_path)

    label_processor.load_label()
    all_label_list = label_processor.scene_attrib
    
    print("train word2vec")
    model = Word2Vec(vector_size=374, window=3, min_count=1, workers=4)
    model.build_vocab(all_label_list, progress_per=10000)
    model.train(all_label_list, total_examples=model.corpus_count, epochs=30, report_delay=1)

    print('done!')
    
    model.save('word2vec_saved.model')
    
    print("save has been done!")

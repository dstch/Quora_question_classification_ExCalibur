import gensim
import pandas as pd


class embbedding():
    def __init__(self):
        self.model_file = 'glove.840B.300d/glove.840B.300d.txt'

    def creat_vector_data(self, csv_file):
        # load model
        model = gensim.models.KeyedVectors.load_word2vec_format(self.model_file)
        data = pd.read_csv(csv_file)
        for sentence in data['question_text']
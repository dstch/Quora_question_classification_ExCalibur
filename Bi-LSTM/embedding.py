import gensim
import pandas as pd


class embbedding():
    def __init__(self):
        self.model_file = 'glove.840B.300d/glove.840B.300d.txt'

    def creat_vector_data(self, csv_file, save_csv_file):
        # load model
        model = gensim.models.KeyedVectors.load_word2vec_format(self.model_file)
        data = pd.read_csv(csv_file)
        vectors = []
        for sentence in data['question_text']:
            for index, word in enumerate(sentence.split()):
                if index == 0:
                    vector = model[word]
                else:
                    vector += model[word]
            vectors.append(vector / len(sentence.split()))
        id_df = data[['qid', 'target']]
        vector_df = pd.DataFrame(vectors, columns=['question_text'])
        result = id_df.join(vector_df)
        result.to_csv(save_csv_file, index=False)



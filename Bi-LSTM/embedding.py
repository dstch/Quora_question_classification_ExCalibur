import gensim
import pandas as pd
import shutil


class embbedding():
    def __init__(self):
        self.model_file = 'GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
        self.save_model_file = 'glove.840B.300d/glove_model.txt'

    def getFileLineNums(self, filename):
        f = open(filename, 'r', encoding='utf-8')
        count = 0
        for line in f:
            count += 1
        f.close()
        return count

    def prepend_line(self, infile, outfile, line):
        with open(infile, 'r', encoding='utf-8') as old:
            with open(outfile, 'w', encoding='utf-8') as new:
                new.write(str(line) + "\n")
                shutil.copyfileobj(old, new)

    def creat_vector_data(self, csv_file, save_csv_file):
        # load model
        model = gensim.models.KeyedVectors.load_word2vec_format(self.model_file, binary=True)
        data = pd.read_csv(csv_file)
        vectors = []
        for sentence in data['question_text']:
            vector = None
            for index, word in enumerate(sentence.split()):
                if index == 0 or len(vector) == 0:
                    try:
                        vector = model[word]
                    except:
                        print(word, 'is not in dictionary by ', sentence)
                else:
                    try:
                        vector = vector + model[word]
                    except:
                        print(word, 'is not in dictionary by ', sentence)
            vectors.append(vector / len(sentence.split()))
        id_df = data[['qid', 'target']]
        vector_df = pd.DataFrame(vectors, columns=['question_text'])
        result = id_df.join(vector_df)
        result.to_csv(save_csv_file, index=False)


if __name__ == '__main__':
    em = embbedding()
    # num_lines = em.getFileLineNums(em.model_file)
    # em.prepend_line(em.model_file, em.save_model_file, num_lines)
    em.creat_vector_data('../train_data/train.csv', '../train_data/train_vector.csv')

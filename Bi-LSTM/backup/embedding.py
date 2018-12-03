import gensim, re, string
import pandas as pd
import shutil, jieba
import numpy


class embbedding():
    def __init__(self):
        self.model_file = 'GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
        self.save_model_file = 'glove.840B.300d/glove_model.txt'
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            self.stopwords = []
            for word in f.readlines():
                self.stopwords.append(word.replace('\n', ''))

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

    def clean_sentence(self, sentence):
        clean_sentence = []
        for word in jieba.cut(sentence):
            if word not in self.stopwords and word not in string.punctuation and word.strip() != '':
                clean_sentence.append(word)
        return clean_sentence

    def sentence_split(self, sentence):
        sentence_string = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+'", " ", sentence)
        return sentence_string.split()

    def creat_vector_data(self, csv_file, save_csv_file):
        # load model
        model = gensim.models.KeyedVectors.load_word2vec_format(self.model_file, binary=True)
        data = pd.read_csv(csv_file)
        vectors = []
        for sentence in data['question_text']:
            vector = numpy.array([0] * 300)
            for index, word in enumerate(self.clean_sentence(sentence)):
                try:
                    vector = vector + model[word]
                except:
                    print(word, 'is not in dictionary by', sentence)
            vectors.append(','.join(vector / len(sentence.split())))
        id_df = data[['qid', 'target']]
        vector_df = pd.DataFrame(vectors, columns=['question_text'])
        result = id_df.join(vector_df)
        result.to_csv(save_csv_file, index=False)


if __name__ == '__main__':
    em = embbedding()
    # num_lines = em.getFileLineNums(em.model_file)
    # em.prepend_line(em.model_file, em.save_model_file, num_lines)
    em.creat_vector_data('../train_data/train.csv', '../train_data/train_vector.csv')
    # for word in jieba.cut('dictionary by Is it ok for brother and sister to have sex?'):
    #     if word not in em.stopwords and word not in string.punctuation and word.strip() != '':
    #         print(word)

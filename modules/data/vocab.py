import numpy as np


class Vocab:

    def __init__(self,
                 vocab_file_path: str = None,
                 glove_file_path: str = None,
                 embedding_size: int = None):
        self.word_list = ['<pad>', '<unk>', '<s>', '<\s>']
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embeddings = None

        self.vocab_file_path = vocab_file_path
        self.glove_file_path = glove_file_path
        self.embedding_size = embedding_size

        if self.vocab_file_path is not None:
            self.add_vocab(self.vocab_file_path)

        if self.glove_file_path and self.embedding_size:
            self.add_embedding(self.glove_file_path, self.embedding_size)

    def get_embedding(self):
        return self.embeddings

    def __getitem__(self, key):
        try:
            return self.w2i[key]
        except KeyError:
            return self.w2i['<unk>']

    def add_vocab(self,
                  vocab_file: str = "./data/finished_files/vocab"):
        with open(vocab_file, "r") as f:
            for line in f:
                # only want the word, not the count
                # Append word and ignore word counter
                self.word_list.append(line.split()[0])

        info = "Read %d words from vocab file path %s" % (
            len(self.word_list), vocab_file)
        print(info)

        # Init mapping index2word and word2index with loaded order
        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1

    def add_embedding(self,
                      glove_file_path: str = "./data/finished_files/glove.6B/glove.6B.100d.txt",
                      embed_size: int = 100):
        print("Loading glove embeddings from path %s with embedding size %s " %
              (glove_file_path, embed_size))

        with open(glove_file_path, 'r') as f:
            model = {}
            w_set = set(self.word_list)
            embedding_matrix = np.zeros(
                shape=(len(self.word_list), embed_size))

            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in w_set:  # only extract embeddings in the word_list
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                    embedding_matrix[self.w2i[word]] = embedding
                    if len(model) % 1000 == 0:
                        print("Loaded %d embedding data" % len(model))

        self.embeddings = embedding_matrix
        info_str = "%d words out of %d has embeddings in the glove file" % \
            (len(model), len(self.word_list))
        print(info_str)

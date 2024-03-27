import pickle
import random
import os
import numpy as np
from typing import Any, List
import pandas as pd
from torch.utils.data import Dataset
from utils.processing import preprocess_text, tokenize_text

from .vocab import Vocab


class Document:
    def __init__(self,
                 id: str,
                 sentences: List[List[str]],
                 tokenized_content: Any = None,
                 tokenized_summary: Any = None):
        self.sentences = sentences
        self.tokenized_content = tokenized_content
        self.tokenized_summary = tokenized_summary
        self.id = id

    def get_sentences(self, indexes):
        merge_sentences = []
        for idx in indexes:
            sent = " ".join(self.sentences[idx])
            sent = sent.strip()
            merge_sentences.append(sent)
        return merge_sentences

    def get_summary(self):
        summary = []
        for sum_content in self.tokenized_summary:
            sent = " ".join(sum_content)
            sent = sent.strip()
            summary.append(sent)
        return summary


class CNNDailyMailDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame = None,
                 path_to_df: str = None,
                 word2id: Vocab = None):
        if df is not None:
            self._df = df
        elif path_to_df is not None:
            self._df = pd.read_csv(path_to_df)

        self.word2id = word2id

    def __len__(self):
        return len(self._df)

    def __call__(self, batch_size, shuffle=True):
        max_len = len(self)
        if shuffle:
            np.random.shuffle(self._df.values)

        batchs = []
        for index in range(0, max_len, batch_size):
            batch = []
            for idx in range(index, index + batch_size, 1):
                batch.append(self.__getitem__(idx))
            yield batch
        #     batchs.append(batch)
        # return batchs

    def __getitem__(self, idx):
        sample = self._df.iloc[idx]
        article, summary = sample["article"], sample["highlights"]

        # Tokenize article and summary
        preprocessed_article = preprocess_text(article)
        tokenized_article = tokenize_text(preprocessed_article)

        preprocessed_summary = preprocess_text(sample["highlights"])
        tokenized_summary = tokenize_text(preprocessed_summary)

        # Word to index to vocab
        max_len = -1
        for sentence in tokenized_article:
            max_len = max(max_len, len(sentence))

        article_array = []
        for sentence in tokenized_article:
            sent = [self.word2id[word.lower()] for word in sentence]
            if len(sent) == 0:
                continue

            # this is to pad at the end of each sequence
            sent += [0 for _ in range(max_len - len(sent))]
            article_array.append(sent)

        article_array = np.array(article_array)
        data = Document(sentences=tokenized_article,
                        tokenized_summary=tokenized_summary,
                        tokenized_content=article_array,
                        id=sample["id"])
        return data


class BatchDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True):
        # assert isinstance(dataset, Dataset)
        assert len(dataset) >= batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.dataset) / self.batch_size) +1 

    def __iter__(self):
        return iter(self.dataset(self.batch_size, self.shuffle))


class PickleReader:
    """
    this class intends to read pickle files converted by RawReader
    """

    def __init__(self, pickle_data_dir="./data/CNN_DM_pickle_data/"):
        """
        :param pickle_data_dir: the base_dir where the pickle data are stored in
        this dir should contain train.p, val.p, test.p, and vocab.p
        this dir should also contain the chunked_data folder
        """
        self.base_dir = pickle_data_dir

    def data_reader(self, dataset_path):
        """
        :param dataset_path: path for data.p
        :return: data: Dataset objects (contain Document objects with doc.content and doc.summary)
        """
        with open(dataset_path, "rb") as f:
            data = pickle.load(f, encoding='bytes')
        return data

    def full_data_reader(self, dataset_type="train"):
        """
        this method read the full dataset
        :param dataset_type: "train", "val", or "test"
        :return: data: Dataset objects (contain Document objects with doc.content and doc.summary)
        """
        return self.data_reader(self.base_dir + dataset_type + ".p")

    def chunked_data_reader(self, dataset_type="train", data_quota=-1):
        """
        this method reads the chunked data in the chunked_data folder
        :return: a iterator of chunks of datasets
        """
        data_counter = 0
        # chunked_dir = self.base_dir + "chunked/"
        chunked_dir = os.path.join(self.base_dir, 'pickled')
        os_list = os.listdir(chunked_dir)
        if data_quota == -1:  # none-quota randomize data
            random.seed()
            random.shuffle(os_list)

        for filename in os_list:
            if filename.startswith(dataset_type):
                # print("filename:", filename)
                chunk_data = self.data_reader(
                    os.path.join(chunked_dir, filename))
                if data_quota != -1:  # cut off applied
                    quota_left = data_quota - data_counter
                    # print("quota_left", quota_left)
                    if quota_left <= 0:  # no more quota
                        break
                    # return partial data
                    elif quota_left > 0 and quota_left < len(chunk_data):
                        yield Dataset(chunk_data[:quota_left])
                        break
                    else:
                        data_counter += len(chunk_data)
                        yield chunk_data
                else:
                    yield chunk_data
            else:
                continue

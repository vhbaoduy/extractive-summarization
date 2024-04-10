from utils.config import Configs, ModelConfig
from modules.data.vocab import Vocab
from modules.models.HER import HERExtractor
from typing import Any, Union
import torch

from utils.processing import preprocess_text, tokenize_text
from nltk.tokenize import sent_tokenize
import numpy as np


def get_extractor(config: ModelConfig,
                  vocab_size: int,
                  embedding_dim: int,
                  pretrained_embeddings: Any):
    extractor = None
    if config.extractor == "HER":
        extractor = HERExtractor(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            word_input_size=config.word_input_size,
            sentence_input_size=config.sentence_input_size,
            decode_hidden_units=config.decode_hidden_units,
            dropout_p=config.dropout,
            num_lstm_layers=config.num_lstm_layers,
            word_lstm_hidden_units=config.word_lstm_hidden_units,
            sentence_lstm_hidden_units=config.sentence_lstm_hidden_units,
            word_pretrained_embedding=pretrained_embeddings
        )
    return extractor


class InferEngine:
    def __init__(self,
                 configs: ModelConfig,
                 vocab_file: str,
                 glove_file: str,
                 embedding_size: str,
                 pretrained_model: str,
                 max_sentences: int = 3):
        self.configs = configs
        self.pretrained_model = pretrained_model
        self.vocab_file = vocab_file
        self.glove_file = glove_file
        self.embedding_size = embedding_size
        self.max_sentences = max_sentences
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vocab = Vocab(vocab_file_path=vocab_file,
                           glove_file_path=glove_file,
                           embedding_size=embedding_size,
                           progress=False)

        self.extractor: torch.nn.Module = None
        self._load_extractor()

    def set_max_num_sentences(self, num: int):
        self.max_sentences = num

    def _load_extractor(self):
        vocab_embeddings = self.vocab.get_embedding()
        vocab_size, emb_dims = vocab_embeddings.shape
        try:
            extractor = torch.load(self.pretrained_model,
                                   map_location=lambda storage, loc: storage)
            extractor.to(self.device)
        except:
            extractor = get_extractor(self.configs,
                                      vocab_size=vocab_size,
                                      embedding_dim=emb_dims,
                                      pretrained_embeddings=vocab_embeddings)
            extractor.load_state_dict(torch.load(self.pretrained_model))
            extractor.to(self.device)
        print(extractor)
        self.extractor = extractor

    def _preprocess(self, content: str):
        preprocessed_content = preprocess_text(content)
        tokenized_content = tokenize_text(preprocessed_content)
        sentences = sent_tokenize(content)

        word2index_array = []
        # Word to index to vocab
        max_len = -1
        for sentence in tokenized_content:
            max_len = max(max_len, len(sentence))

        for sentence in tokenized_content:
            sent = [self.vocab[word.lower()] for word in sentence]
            if len(sent) == 0:
                continue

            # this is to pad at the end of each sequence
            sent += [0 for _ in range(max_len - len(sent))]
            word2index_array.append(sent)

        word2index_array = np.array(word2index_array)
        return sentences, tokenized_content, word2index_array

    def summarize(self,
                  content: str,
                  max_num_of_sentences: int = None):
        sentences, _, word_vectors = self._preprocess(content)
        if max_num_of_sentences is None:
            max_num_of_sentences = self.max_sentences

        if max_num_of_sentences == -1:
            max_num_of_sentences = int(len(sentences) // 3)

        if max_num_of_sentences >= len(sentences):
            max_num_of_sentences = len(sentences)

        if min(word_vectors.shape) == 0:
            return sentences

        summary_index = None
        if len(sentences) < max(self.configs.kernel_sizes):
            summary_index = range(len(sentences))
        else:
            # Extract document to get summary index
            sents = torch.from_numpy(word_vectors).to(self.device)
            probs = self.extractor(sents)
            probs = probs.detach().cpu().numpy()

            summary_index = np.argsort(np.reshape(
                probs, len(probs)))[-max_num_of_sentences:]
            summary_index = summary_index[::-1]
            summary_index = sorted(summary_index)

        # Return summary sentence
        summary_sents = sentences
        if summary_index is not None:
            summary_sents = [sentences[idx].replace(
                "\n", " ") for idx in summary_index]
        return summary_sents

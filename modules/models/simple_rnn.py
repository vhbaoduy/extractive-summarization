import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

from typing import Any, Union
import numpy as np

torch.manual_seed(233)


class SimpleRNN(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 word_input_size: int,
                 sentence_input_size: int,
                 word_lstm_hidden_units: int,
                 sentence_lstm_hidden_units: int,
                 dropout_p: float = 0.1,
                 vocab_size: int = 200000,
                 word_pretrained_embedding: Union[np.ndarray, Any] = None,
                 sentence_rep: bool = False,
                 **kwargs):
        """
        Args:
            embedding_dim (int): _description_
            position_size (int): _description_
            word_input_size (int): _description_
            sentence_input_size (int): _description_
            word_lstm_hidden_units (int): _description_
            sentence_lstm_hidden_units (int): _description_
            dropout_p (float, optional): _description_. Defaults to 0.1.
            vocab_size (int, optional): _description_. Defaults to 200000.
            word_pretrained_embedding (Any, optional): _description_. Defaults to None.
            sentence_rep (bool, optional): _description_. Defaults to False.
        """
        super(SimpleRNN, self).__init__()

        # Init paramters
        self.dropout_p = dropout_p
        self.embedding_dim = embedding_dim
        self.word_input_size = word_input_size
        self.sentence_input_size = sentence_input_size
        self.word_lstm_hidden_units = word_lstm_hidden_units
        self.sentence_lstm_hidden_units = sentence_lstm_hidden_units
        self.vocab_size = vocab_size
        self.word_pretrained_embedding = word_pretrained_embedding
        self.sentence_rep = sentence_rep

        # Init network for sentence/word extraction
        self.word_embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                           embedding_dim=self.embedding_dim)
        if self.word_pretrained_embedding is not None:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(word_pretrained_embedding)
            )

        self.word_lstm_layer = nn.LSTM(
            input_size=self.word_input_size,
            hidden_size=self.word_lstm_hidden_units,
            num_layers=2,
            dropout=self.dropout_p,
            batch_first=True,
            bidirectional=True)

        self.sentence_lstm_layer = nn.LSTM(
            input_size=self.sentence_input_size,
            hidden_size=self.sentence_lstm_hidden_units,
            num_layers=2,
            dropout=self.dropout_p,
            batch_first=True,
            bidirectional=True)

        # Need to define linear node
        if self.sentence_rep:
            self.decoder = nn.Sequential(nn.Linear(self.sentence_lstm_hidden_units * 4, 100),
                                         nn.Tanh(),
                                         nn.Linear(100, 1),
                                         nn.Sigmoid())
        else:
            self.decoder = nn.Sequential(nn.Linear(self.sentence_lstm_hidden_units * 2, 100),
                                         nn.Tanh(),
                                         nn.Linear(100, 1),
                                         nn.Sigmoid())
        self.dropout_layer = nn.Dropout(p=self.dropout_p)


    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)

        return torch.cat(result, dim=0)

    # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
    # Input: list to tokens
    def forward(self, x: torch.Tensor):
        sequence_length = torch.sum(torch.sign(
            x), dim=1).data  # ex.=[3,2]-> size=2

        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # word level LSTM
        # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        word_features = self.word_embedding(x)
        # word_features = self.drop(word_features)

        # output: word_outputs (N,W,h)
        word_outputs, _ = self.word_lstm_layer(word_features)
        sent_features = self._avg_pooling(word_outputs, sequence_length)

        sent_features = sent_features.view(1,
                                           sequence_num,
                                           self.sentence_input_size)  # output:(1,N,h)

        sent_features = self.dropout_layer(sent_features)

        # sentence level LSTM
        enc_output, _ = self.sentence_lstm_layer(sent_features)
        enc_output = self.dropout_layer(enc_output)

        if self.sentence_rep:
            doc_features = torch.mean(enc_output, dim=1, keepdim=True)
            doc_features = doc_features.expand(enc_output.size())
            enc_output = torch.cat([enc_output, doc_features], dim=-1)

        prob = self.decoder(enc_output)
        prob = prob.view(sequence_num, 1)

        return prob

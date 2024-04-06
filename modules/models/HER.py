import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F

from typing import Any, Union
import numpy as np

torch.manual_seed(233)


class HERExtractor(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 word_input_size: int,
                 sentence_input_size: int,
                 word_lstm_hidden_units: int,
                 sentence_lstm_hidden_units: int,
                 decode_hidden_units: int = 200,
                 num_lstm_layers: int = 2,
                 kernel_sizes: list = [1, 2, 3],
                 num_filters: int = 50,
                 dropout_p: float = 0.1,
                 vocab_size: int = 200000,
                 word_pretrained_embedding: Union[np.ndarray, Any] = None,
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
        super(HERExtractor, self).__init__()

        # Init paramters
        self.dropout_p = dropout_p
        self.embedding_dim = embedding_dim
        self.word_input_size = word_input_size
        self.num_lstm_layers = num_lstm_layers
        self.sentence_input_size = sentence_input_size
        self.word_lstm_hidden_units = word_lstm_hidden_units
        self.sentence_lstm_hidden_units = sentence_lstm_hidden_units
        self.vocab_size = vocab_size
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.word_pretrained_embedding = word_pretrained_embedding
        self.decode_hidde_units = decode_hidden_units

        # Init network for sentence/word extraction

        # Word embedding from glove
        self.word_embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                           embedding_dim=self.embedding_dim)
        if self.word_pretrained_embedding is not None:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(word_pretrained_embedding)
            )

        # Word-level encode layer
        self.word_lstm_layer = nn.LSTM(
            input_size=self.word_input_size,
            hidden_size=self.word_lstm_hidden_units,
            num_layers=self.num_lstm_layers,
            dropout=self.dropout_p,
            batch_first=True,
            bidirectional=True)

        # Sentence-level encode layer
        self.sentence_lstm_layer = nn.LSTM(
            input_size=self.sentence_input_size,
            hidden_size=self.sentence_lstm_hidden_units,
            num_layers=self.num_lstm_layers,
            dropout=self.dropout_p,
            batch_first=True,
            bidirectional=True)

        # Local feature with CNN
        self.conv_layers = []
        for ksz in self.kernel_sizes:
            self.conv_layers.append(nn.Conv2d(in_channels=1,
                                              out_channels=self.num_filters,
                                              kernel_size=(ksz, 2*self.sentence_lstm_hidden_units)))
        self.conv_layers = nn.ModuleList(self.conv_layers)

        # Sentence decoder
        self.input_decode_dim = 2*self.num_lstm_layers * self.sentence_lstm_hidden_units + \
            len(self.kernel_sizes) * self.num_filters
        self.decoder1 = nn.Sequential(nn.Linear(self.input_decode_dim, self.decode_hidde_units),
                                      nn.Tanh(),
                                      nn.Linear(self.decode_hidde_units, 1),
                                      nn.Sigmoid())

        self.decoder2 = nn.Sequential(nn.Linear(self.input_decode_dim, self.decode_hidde_units),
                                      nn.Tanh(),
                                      nn.Linear(self.decode_hidde_units, 1),
                                      nn.Sigmoid())

        # self.decoder2 = nn.Sequential(nn.Linear())

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

        # h = 2* num_layers * hidden_units (400)

        # output: word_outputs (N,W,h)
        word_outputs, _ = self.word_lstm_layer(word_features)
        sent_features = self._avg_pooling(word_outputs, sequence_length)

        # Output: [1, N_sents, h]
        sent_outputs = sent_features.view(1,
                                          sequence_num,
                                          self.sentence_input_size)  # output:(1,N,h)

        sent_outputs = self.dropout_layer(sent_outputs)

        # sentence level LSTM
        # Output: [1, N_sents, h]
        sent_embeddings, _ = self.sentence_lstm_layer(sent_outputs)
        sent_embeddings = self.dropout_layer(sent_embeddings)

        # Compute global feature
        # Ouput: [1, h]
        doc_features = torch.mean(sent_embeddings, dim=1)

        # Compute local feature
        # Expand dim in sentence embeddings
        sent_emb_temp = torch.unsqueeze(sent_embeddings, 1)
        local_outputs = []
        for conv2d in self.conv_layers:
            out = conv2d(sent_emb_temp)
            out = F.relu(out)  # [1, num_filters, num_sentence, 1]
            out = torch.squeeze(out, 3)
            out = F.max_pool1d(out, out.size(2))  # [1, num_filters]
            out = torch.squeeze(out, 2)
            local_outputs.append(out)

        # Update sentence representations by concat feats
        sent_embeddings = torch.squeeze(sent_embeddings, 0)
        doc_outputs = torch.squeeze(
            doc_features, 0).expand(sent_embeddings.size())
        local_outputs_tmp = [torch.squeeze(local_output, 0).expand((sequence_num, local_output.size()[-1]))
                             for local_output in local_outputs]

        enc_output = torch.cat([
            sent_embeddings,
            doc_outputs,
            *local_outputs_tmp,
        ], dim=-1)
        mu_1 = self.decoder1(enc_output)
        mu_2 = self.decoder2(torch.multiply(enc_output, 1-mu_1))

        prob = (mu_1 + mu_2)/2
        prob = prob.view(sequence_num, 1)

        return prob

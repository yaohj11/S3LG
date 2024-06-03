import torch
import torch.nn as nn
from torch import Tensor

from model.util import generate_mask_with_seq_length

from .transformer_layers import PositionalEncoding, TransformerEncoderLayer


class Encoder(nn.Module):
    """
    Base encoder class
    """
    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size





class EncoderTransformer(Encoder):
    """
    Transformer Encoder
    """
    def __init__(self, input_size, vocab_size=-1, emb_dropout=0.0, hidden_size=512, num_layers=2, 
                 num_heads=4, ff_size=2048, dropout=0.3, num_classes=-1):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        """
        super(EncoderTransformer, self).__init__()

        self.vocab_size = vocab_size
        self.input_size = input_size
        if vocab_size != -1:
            self.embedding = nn.Embedding(vocab_size, input_size)
        self.dropout_emb = nn.Dropout(p=emb_dropout, inplace=True)

        self.flag_convert_emb = input_size != hidden_size
        if self.flag_convert_emb:
            self.convert_emb = nn.Sequential(
                                nn.Conv1d(input_size, hidden_size, kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm1d(hidden_size),
                                nn.ReLU(inplace=True),
                            )

        # build all (num_layers) layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.num_classes = num_classes
        if num_classes != -1:
            self.final_fc = nn.Linear(hidden_size, num_classes)

        self._output_size = hidden_size

    def forward(self, src_feats, src_length):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        Args:
            - src_feats: (batch_size, max_src_len) or (batch_size, max_src_len, input_size)
            - src_length: length of src inputs(counting tokens before padding), 
                          shape (batch_size)
        Returns:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        mask = generate_mask_with_seq_length(src_length)
        mask = mask.unsqueeze(1)

        if self.vocab_size != -1:
            src_feats = self.embedding(src_feats)
        x = self.pe(src_feats)  # add position encoding to word embeddings
        x = self.emb_dropout(x)

        if self.flag_convert_emb:
            embed_src = self.convert_emb(embed_src.transpose(1, 2)).transpose(1, 2)

        for layer in self.layers:
            x = layer(x, mask)
        outputs = self.layer_norm(x)
        
        if self.num_classes != -1:
            outputs_fc = self.final_fc(outputs)
            return [outputs, outputs_fc], None, src_length

        return [outputs,], None, src_length

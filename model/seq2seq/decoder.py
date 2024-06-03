import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.eval import Beamer, process_seq_for_slt
from model.util import generate_mask_with_seq_length, generate_subsequent_mask
from torch import Tensor


from .transformer_layers import PositionalEncoding, TransformerDecoderLayer


# pylint: disable=abstract-method
class Decoder(nn.Module):
    """
    Base decoder class
    """

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)

        :return:
        """
        return self._output_size



class DecoderTransformer(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self, vocab_size, emb_size, num_layers=2, num_heads=8, hidden_size=512,
                 ff_size=2048, dropout=0.1, emb_dropout=0.1):
        """
        Initialize a Transformer decoder.

        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(DecoderTransformer, self).__init__()

        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self._output_size = vocab_size

        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList([TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout) for _ in range(num_layers)])

        self.pe = PositionalEncoding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)


    def forward(self, input_seq, encoder_outputs, src_lens, trg_lens, decoder_hidden=None, prev_att_vector=None):
        """
        Transformer decoder forward pass.

        Args:
            - input_seq

        """
        src_mask = generate_mask_with_seq_length(src_lens).unsqueeze(1)
        trg_mask = generate_mask_with_seq_length(trg_lens).unsqueeze(1)

        trg_embed = self.embedding(input_seq) # (B, T, D)

        x = self.pe(trg_embed)  # add position encoding to word embedding
        x = self.emb_dropout(x)
        trg_mask = trg_mask & generate_subsequent_mask(
            trg_embed.size(1)).type_as(trg_mask)

        for layer in self.layers:
            x = layer(x=x, memory=encoder_outputs,
                      src_mask=src_mask, trg_mask=trg_mask)

        x = self.layer_norm(x)
        output = self.output_layer(x)

        trg_mask = generate_mask_with_seq_length(trg_lens)

        return [output, trg_mask]

    def forward_train_Text(self, BOS, encoder_outputs, encoder_hidden, len_input, label, len_label, teacher_forcing=0.0):
        batch_size = encoder_outputs.size(0)
        BOS = label.new_full(size=(batch_size, 1), fill_value=BOS)
        label = torch.cat((BOS, label), dim=1)[:, :-1]
        
        output = self.forward(label, encoder_outputs, len_input, len_label)
        return output

    def forward_eval_Text(self, BOS, EOS, BLK, encoder_outputs, encoder_hidden, len_input, beam_size, alpha=0.0, max_seq_len=50):
        batch_size = encoder_outputs.size(0)

        input_seq = len_input.new_full(size=(batch_size, beam_size, 1), fill_value=BOS)
        outputs = encoder_outputs.new_zeros((batch_size, beam_size, self.vocab_size))
        len_output = len_input.new_full(size=(batch_size,), fill_value=1)
        beamer = Beamer(beam_size, batch_size, max_seq_len, EOS, BLK, encoder_outputs.device, alpha)

        entropy=[]
        for i in range(max_seq_len):
            for j in range(beam_size):
                decoder_output = self.forward(
                    input_seq[:, j, :], encoder_outputs, len_input, len_output)
                outputs[:, j, :] = decoder_output[0][:, -1, :]
                dist=torch.distributions.Categorical(torch.softmax(decoder_output[0][:, -1, :],dim=1))
                entropy.append(dist.entropy().unsqueeze(1))
            outputs = F.log_softmax(outputs, dim=-1)
            next_word, pos = beamer.beam_search(outputs, i)
            # prev_att_vector = prev_att_vector.gather(1, pos.unsqueeze(-1).expand_as(prev_att_vector))
            # hiddens = hiddens.gather(2, pos.unsqueeze(0).unsqueeze(-1).expand_as(hiddens))
            input_seq = input_seq.gather(1, pos.unsqueeze(-1).expand_as(input_seq))
            input_seq = torch.cat((input_seq, next_word.unsqueeze(-1)), dim=-1)
            len_output += 1

        beamer.sort()
        pred_seqs = beamer.final_seqs[:, :, 0]
        
        entropy=torch.cat(entropy,dim=1)
        max_entropy=[]

        processed_pred_seqs = []
        for i in range(batch_size):
            pred = process_seq_for_slt(pred_seqs[:, i].tolist(), end_token=EOS, mode='first')
            processed_pred_seqs.append(pred)
            entropyi=torch.mean(entropy[i,:len(pred)])+0
            max_entropy.append(entropyi)

        # entropy=torch.cat(entropy,dim=1)
        # entropy=torch.mean(entropy,dim=1)
        return processed_pred_seqs,max_entropy

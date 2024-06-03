import numpy as np
import torch
import torch.nn.functional as F


class Beamer(object):

    def __init__(self, beam_size, batch_size, max_len, end_token, blank_token):
        '''
        seqs:   [T, B, beam]
        scores: [B, beam]
        '''
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.pointer = torch.zeros(batch_size).long()
        self.beam_seqs = torch.LongTensor(max_len, batch_size, beam_size).fill_(blank_token).cuda()
        self.beam_scores = torch.zeros(batch_size, beam_size).cuda()
        self.beam_scores.fill_(float('-inf'))
        self.beam_scores[:, 0] = 0
        
        self.final_seqs = torch.LongTensor(max_len, batch_size, beam_size).fill_(blank_token).cuda()
        self.final_scores = torch.zeros(batch_size, beam_size).cuda()
        self.end_token = end_token

    def beam_search(self, output, time_stamp):
        """
        output(decoder): [B, beam, W]
        hiddens: [n, B, beam, D]
        """
        assert time_stamp < self.max_len

        # pick out top-beam from each seq => [B, beam, beam]
        output_score, output_seq = torch.topk(output, self.beam_size, dim=-1)

        # [B, (beam*beam)]
        self.beam_scores = self.beam_scores.unsqueeze(2).expand_as(output_score).contiguous()
        self.beam_scores = (self.beam_scores + output_score).view(self.batch_size, -1)

        # sort result from beam*beam seqs => [B, beam]
        self.beam_scores, pos = torch.topk(self.beam_scores, self.beam_size, dim=-1)

        # next input: [B, beam]
        input_seq = output_seq.view(self.batch_size, -1).gather(-1, pos)

        # sort seqs and hiddens
        pos = pos // self.beam_size
        self.beam_seqs = self.beam_seqs.gather(2, pos.unsqueeze(0).expand_as(self.beam_seqs))
        self.beam_seqs[time_stamp] = input_seq

        for i in range(self.batch_size):
            for j in range(self.beam_size):
                if self.pointer[i] >= self.beam_size:
                    break
                if input_seq[i, j] == self.end_token or time_stamp == self.max_len-1:
                    self.final_seqs[:, i, self.pointer[i]] = self.beam_seqs[:, i, j]
                    self.final_scores[i, self.pointer[i]] = self.beam_scores[i, j]
                    self.beam_scores[i, j] = float('-inf')
                    self.pointer[i] += 1
                # delete seq which has the same word [lead to poor performance]
                # if input_seq[i, j] == self.beam_seqs[time_stamp-1, i, j] and time_stamp != 0:
                #     self.beam_scores[i, j] = float('-inf')

        return  input_seq, pos

    def sort(self):
        self.final_scores, indices = torch.sort(self.final_scores, dim=1, descending=True)
        self.final_seqs = self.final_seqs.gather(2, indices.unsqueeze(0).expand_as(self.final_seqs))


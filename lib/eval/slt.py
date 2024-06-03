import numpy as np
import torch
import torch.nn.functional as F
from lib.metric import compute_bleu, compute_rouge


class SLT_Eval(object):
    def __init__(self, vocab):
        self.vocab = vocab
    
    def eval_batch(self, ref_list, hyp_list):
        hyp = [[self.vocab[y] for y in x] for x in hyp_list]
        ref = [[self.vocab[y] for y in x] for x in ref_list]
        a = compute_rouge([' '.join(x) for x in hyp], [' '.join(x) for x in ref])
        
        ref = [[x,] for x in ref]
        b1, _, _, _, _, _ = compute_bleu(ref, hyp, max_order=1, smooth=False)
        b2, _, _, _, _, _ = compute_bleu(ref, hyp, max_order=2, smooth=False)
        b3, _, _, _, _, _ = compute_bleu(ref, hyp, max_order=3, smooth=False)
        b4, _, _, _, _, _ = compute_bleu(ref, hyp, max_order=4, smooth=False)


        # ref = [' '.join(x) for x in ref]
        # hyp = [' '.join(x) for x in hyp]
        # b = compute_bleu(ref, hyp)
        # b1 =b['bleu1']/100.0
        # b2 =b['bleu2']/100.0
        # b3 =b['bleu3']/100.0
        # b4 =b['bleu4']/100.0
        # return a*100, [b1*100, b2*100, b3*100, b4*100]

        return a['rouge_l/f_score']*100, [b1*100, b2*100, b3*100, b4*100]


class Beamer(object):

    def __init__(self, beam_size, batch_size, max_len, end_token, blank_token, device, alpha=0.0):
        '''
        seqs:   [T, B, beam]
        scores: [B, beam]
        '''
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.pointer = torch.zeros(batch_size).long()
        self.beam_seqs = torch.LongTensor(max_len, batch_size, beam_size).fill_(blank_token).to(device)
        self.beam_scores = torch.zeros(batch_size, beam_size).to(device)
        self.beam_scores.fill_(float('-inf'))
        self.beam_scores[:, 0] = 0
        
        self.final_seqs = torch.LongTensor(max_len, batch_size, beam_size).fill_(blank_token).to(device)
        self.final_scores = torch.zeros(batch_size, beam_size).to(device)
        self.end_token = end_token

        self.alpha=alpha

    def beam_search(self, output, time_stamp):
        """
        output(decoder): [B, beam, W]
        # hiddens: [n, B, beam, D]
        pos : [B, Beam]
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

        length_penalty = ((5.0 + (time_stamp + 1)) / 6.0) ** self.alpha
        # print(length_penalty, time_stamp, self.alpha)
        for i in range(self.batch_size):
            for j in range(self.beam_size):
                if self.pointer[i] >= self.beam_size:
                    break
                if input_seq[i, j] == self.end_token or time_stamp == self.max_len-1:
                    self.final_seqs[:, i, self.pointer[i]] = self.beam_seqs[:, i, j]
                    self.final_scores[i, self.pointer[i]] = self.beam_scores[i, j] / length_penalty
                    self.beam_scores[i, j] = float('-inf')
                    self.pointer[i] += 1
                # delete seq which has the same word [lead to poor performance]
                # if input_seq[i, j] == self.beam_seqs[time_stamp-1, i, j] and time_stamp != 0:
                #     self.beam_scores[i, j] = float('-inf')

        return  input_seq, pos

    def sort(self):
        self.final_scores, indices = torch.sort(self.final_scores, dim=1, descending=True)
        self.final_seqs = self.final_seqs.gather(2, indices.unsqueeze(0).expand_as(self.final_seqs))

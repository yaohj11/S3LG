import glob
import logging
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import pickle


class InfoToolKit(object):
    def __init__(self,gloss_pth,T, tag, src='word', trg='gloss', load_feature=False, external_corpus=''):
        self.tag = tag
        self.external_corpus = external_corpus

        with open(os.path.join(self.tag, 'text/gloss_map.txt'), 'r') as f:
            self.gloss_map = [x.strip().lower() for x in f]
        with open(os.path.join(self.tag, 'text/word_map.txt'), 'r') as f:
            self.word_map = [x.strip() for x in f]
        self.gloss_map.extend(['<BOS>', '<EOS>', '<BLK>', '<UNK>'])
        self.word_map.extend(['<RULE>', '<MODEL>', '<BLK>', '<UNK>'])

        # self.word_map=self.word_map+self.gloss_map
        self._src_rule = self.word2idx(['<RULE>'] , src)[0]
        self._src_model = self.word2idx(['<MODEL>'] , src)[0]
        self._src_BLK = self.word2idx(['<BLK>'] , src)[0]
        self._src_UNK = self.word2idx(['<UNK>'] , src)[0]
        self._trg_BOS = self.word2idx(['<BOS>'] , trg)[0]
        self._trg_EOS = self.word2idx(['<EOS>'] , trg)[0]
        self._trg_BLK = self.word2idx(['<BLK>'] , trg)[0]
        self._trg_UNK = self.word2idx(['<UNK>'] , trg)[0]

        self.num_glosses = len(self.gloss_map)
        self.num_words = len(self.word_map)

        self.info = []
        if tag != '':
            self.info = self.get_info(load_feature) # [train, dev, test]
        self.info_external_corpus = []
        if external_corpus != '':
            self.info_external_corpus = self.get_external_info(gloss_pth,T,load_feature)

    def word2idx(self, word_list, label_type='gloss', enable_unk=False):
        if label_type == 'gloss':
            label_map = self.gloss_map
        elif label_type == 'word':
            label_map = self.word_map
        if enable_unk:
            indices = []
            for x in word_list:
                if x in label_map:
                    indices.append(label_map.index(x))
                else:
                    indices.append(label_map.index('<UNK>'))
            return indices
        return [label_map.index(x) for x in word_list]

    def idx2word(self, idx_list, label_type='gloss'):
        if label_type == 'gloss':
            label_map = self.gloss_map
        elif label_type == 'word':
            label_map = self.word_map
        return [label_map[x] for x in idx_list]

    def get_info(self, load_feature):
        info = []
        for task in ['train', 'dev', 'test']:
            info_task = []
            with open(os.path.join(self.tag, 'text/{:s}.txt'.format(task)), 'r') as f:
                lines = f.readlines()[1:]
            for x in lines:
                x = x.strip().split('|')
                line = {
                    'name': x[0],
                    'label_gloss': self.word2idx(x[1].lower().split(' '), 'gloss', enable_unk=True),
                    'label_word': self.word2idx(x[2].split(' '), 'word', enable_unk=True),
                }
                info_task.append(line)
            info.append(info_task)
        return info

    def get_external_info(self,gloss_pth,T, load_feature=False):
        info = []
        with open(gloss_pth, 'rb') as f:
            gloss = pickle.load(f)
        with open(self.external_corpus,'r') as f:
            lines=f.readlines()
        num=len(lines)
        
        for x in tqdm(lines):
            x = x.strip().split('|')
            if T==1:
                line = {
                    'name': x[0],
                    'label_gloss':self.word2idx(gloss[x[0]][0],'gloss', enable_unk=True),
                    'label_word': self.word2idx(['<RULE>']+x[2].split(' '), 'word', enable_unk=True),
                    'entropy':gloss[x[0]][1]
                }
            else:
                rule_based=self.word2idx(gloss[x[0]][0][0],'gloss', enable_unk=True)
                rule_entropy=gloss[x[0]][0][1]
                rule_text=self.word2idx(['<RULE>']+x[2].split(' '), 'word', enable_unk=True)
                model_based=gloss[x[0]][1][0]
                model_entropy=gloss[x[0]][1][1]
                model_text=self.word2idx(['<MODEL>']+x[2].split(' '), 'word', enable_unk=True)
                if random.random() < 0.5:
                    choose_gloss=rule_based
                    entropy=model_entropy
                    text=rule_text
                else:
                    choose_gloss=model_based
                    entropy=model_entropy
                    text=model_text
                line = {
                    'name': x[0],
                    'label_gloss': choose_gloss,
                    'label_word': text,
                    'entropy':entropy,
                }
            info.append(line)
        return info

    def __str__(self):
        line = '\n-----InfoKit------\n'
        line += 'num_infos: '+','.join([str(len(x)) for x in self.info])+'\n'
        line += 'num_external_corpus: {:d}\n'.format(len(self.info_external_corpus))
        line +='num_glosses: {:d}, num_words: {:d}\n'.format(self.num_glosses, self.num_words)
        line += 'src_BLK: {:d}, src_UNK: {:d}\n'.format(self._src_BLK, self._src_UNK)
        line += 'trg_BOS: {:d}, trg_EOS: {:d}, trg_BLK: {:d}, trg_UNK: {:d}\n'.format(
            self._trg_BOS, self._trg_EOS, self._trg_BLK, self._trg_UNK
        )
        line += '------------------'
        return line





class FeatDataset(Dataset):
    '''
        Text and Feat
    '''
    def __init__(self, info, trg='gloss', random_unk=[False, None], external_corpus=None, random_throw=False):
        super().__init__()
        self.list = info
        self.trg = trg

        self.random_unk = random_unk[0]
        self.src_unk = random_unk[1]
        self.random_throw = random_throw

        self.external_corpus = external_corpus
        self.training_list = info

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        name=self.list[index]['name']
        label_input = self.list[index]['label_word']
        ps_flag=label_input[0]==3001 or label_input[0]==3002
        tag=label_input.pop(0)
        if self.random_throw and random.random()< 0.6 and ps_flag==True:
            reserve_pos = list(range(len(label_input)))
            num_throw = len(label_input) // 5
            num_throw = random.randint(0, num_throw)
            for _ in range(num_throw):
                idx = random.randint(0, len(reserve_pos) - 1)
                elm=reserve_pos.pop(idx)
                idx = random.randint(0, len(reserve_pos) - 1)
                reserve_pos.insert(idx,elm)
            label_input = np.array(label_input)[reserve_pos].tolist()
        if self.random_throw and random.random()< 0.6 and ps_flag==True:
            reserve_pos = list(range(len(label_input)))
            num_throw = len(label_input) // 5
            num_throw = random.randint(0, num_throw)
            for _ in range(num_throw):
                idx = random.randint(0, len(reserve_pos) - 1)
                reserve_pos.pop(idx)
            label_input = np.array(label_input)[reserve_pos].tolist()
        label_input.insert(0,tag)
        label_output = self.list[index]['label_gloss']
        return name,label_input, label_output, index

    def change_mode(self, mode):
        if mode == 'e':
            self.list = self.external_corpus
        elif mode == 't':
            self.list = self.training_list
        elif mode == 'et':
            self.list = self.training_list + self.external_corpus
        elif mode == 'ret':
            self.list = self.training_list + random.sample(self.external_corpus, len(self.training_list))


class CollateFeat2Text(object):
    def __init__(self, src_BLK, tgt_BLK, tgt_EOS, pad_label=True):
        self.src_BLK = src_BLK
        self.tgt_BLK = tgt_BLK
        self.tgt_EOS = tgt_EOS
        self.flag_pad_label = pad_label

    def collate_fn_feat(self, batch):
        '''          
            Args:
                feat: [(T,C), (T)]      
            Returns:
                text: (B, T)
        '''
        # batch.sort(key=lambda x: x[0].shape[0], reverse=True)
        name,labels,gloss, indices = zip(*batch)

        for i in range(len(gloss)):
            gloss[i].append(self.tgt_EOS)
        len_gloss = [len(x) for x in gloss]
        if self.flag_pad_label:
            pad_gloss = torch.full(size=(len(len_gloss), max(len_gloss)), fill_value=self.tgt_BLK, dtype=torch.long)
            for i, tmp in enumerate(gloss):
                pad_gloss[i, :len_gloss[i]] = torch.LongTensor(tmp)
        len_gloss = torch.LongTensor(len_gloss)

        len_label = [len(x) for x in labels]
        if self.flag_pad_label:
            pad_labels = torch.full(size=(len(len_label), max(len_label)), fill_value=self.src_BLK, dtype=torch.long)
            for i, tmp in enumerate(labels):
                pad_labels[i, :len_label[i]] = torch.LongTensor(tmp)
        else:
            pad_labels = []
            for tmp in labels:
                pad_labels.extend(tmp)
            pad_labels = torch.LongTensor(pad_labels)
        len_label = torch.LongTensor(len_label)

        batch = {
            'name':name,
            'input': pad_labels,
            'len_input': len_label,
            'output': pad_gloss,
            'len_output': len_gloss,
            'index': indices,
        }

        return batch





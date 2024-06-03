import argparse
import logging
import os
import random
import sys
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import alter_path
from data import InfoToolKit, CollateFeat2Text, FeatDataset
from config import load_config
from lib.eval import Beamer, SLT_Eval, process_seq_for_slt
from lib.metric import compute_wer
from lib.train import MaskedCrossEntropyLoss
from lib.util import AverageMeter, init_logging
from model.seq2seq import (DecoderTransformer, EncoderTransformer)

from tqdm import tqdm

def r_div(student, teacher, or_mask):
    mask = or_mask.unsqueeze(-1)
    mask = mask.repeat(1, 1, 1119)
    teacher = F.softmax(teacher, dim=-1)
    teacher = teacher.masked_fill(~mask, 0.0)
    kl_loss = nn.KLDivLoss(reduction="sum")
    student = F.log_softmax(student, dim=-1)
    student = student.masked_fill(~mask, 0.0)
    loss = kl_loss(student, teacher) / or_mask.float().sum()
    return loss

def train_once(epoch,opts, batch,encoder, decoder, encoder_optim, decoder_optim, device, criterion):
    feature = batch['input']
    len_input = batch['len_input']
    label = batch['output']
    len_label = batch['len_output']

    feature = feature.to(device)
    len_input = len_input.to(device)
    label = label.to(device)
    len_label = len_label.to(device)

    encoder.train()
    decoder.train()

    encoder_optim.zero_grad()
    # decoder_optim.zero_grad()

    if epoch < 200:
        #aug1
        encoder_outputs1, encoder_hidden1, len_input1 = encoder(feature, len_input)
        decoder_outputs1 = decoder.forward_train_Text(
            _BOS, encoder_outputs1[0], encoder_hidden1, len_input1, label, len_label, teacher_forcing=opts.teacher_forcing)
        
        #aug2
        encoder_outputs2, encoder_hidden2, len_input2 = encoder(feature, len_input)
        decoder_outputs2 = decoder.forward_train_Text(
            _BOS, encoder_outputs2[0], encoder_hidden2, len_input2, label, len_label, teacher_forcing=opts.teacher_forcing)

        translation_loss1, pred_seqs, num_correct, num_words = criterion(
            decoder_outputs1[0], label, len_label)
        
        translation_loss2, pred_seqs, num_correct, num_words = criterion(
            decoder_outputs2[0], label, len_label)
    
        l_div1 = r_div(decoder_outputs1[0], decoder_outputs2[0], decoder_outputs1[1])
        l_div2 = r_div(decoder_outputs2[0], decoder_outputs1[0], decoder_outputs1[1])
        
        if epoch<10:
            weight=epoch*0.1
            loss = translation_loss1+translation_loss2 + (l_div1 + l_div2) * 20*weight
        else:
            loss = translation_loss1+translation_loss2 + (l_div1 + l_div2) * 20
    else:
        encoder_outputs1, encoder_hidden1, len_input1 = encoder(feature, len_input)
        decoder_outputs1 = decoder.forward_train_Text(
            _BOS, encoder_outputs1[0], encoder_hidden1, len_input1, label, len_label, teacher_forcing=opts.teacher_forcing)
        
        translation_loss1, pred_seqs, num_correct, num_words = criterion(
            decoder_outputs1[0], label, len_label)
        
        loss = translation_loss1
        
    loss.backward()
    encoder_optim.step()

    return loss.item(), pred_seqs.detach(), num_correct, num_words


def eval_once(task,batch, encoder, decoder, device, beam_size=1, max_seq_len=35, alpha=0.0):
    name=batch['name']
    feature = batch['input']
    len_input = batch['len_input']
    label = batch['output']
    len_label = batch['len_output']

    with torch.no_grad():
        feature = feature.to(device)
        len_input = len_input.to(device)

        encoder.eval()
        decoder.eval()

        encoder_outputs, encoder_hidden, len_input = encoder(feature, len_input)

        pred_seqs,entropy = decoder.forward_eval_Text(
            _BOS, _EOS, _BLK, encoder_outputs[0], encoder_hidden, len_input, beam_size, alpha, max_seq_len=max_seq_len)
        
        if task=='add':
            return name,pred_seqs,entropy
        
        err_total = 0
        count = 0
        hyp = []
        ref = []
        for i, tmp in enumerate(label):
            pred = pred_seqs[i]
            hyp.append(pred)
            ref.append(tmp[:len_label[i]-1].tolist())
            err = compute_wer(tmp[:len_label[i]-1], pred)[0]
            err_total += err
            count += 1
    return err_total, count, [hyp, ref]


def eval_iter(opts, task, loader,encoder, decoder, device, eval_tool):
    hyp_list = []
    ref_list = []
    total_err = 0
    total_count = 0
    name=[]
    pred_seqs=[]
    entropy=[]
    if task == 'add':
        for step, batch in enumerate(loader):
            namei,pred_seqsi,entropyi = eval_once(task,batch, encoder, decoder, device, 1, opts.max_seq_len, opts.alpha)
            name.extend(namei)
            pred_seqs.extend(pred_seqsi)
            entropy.extend(entropyi)
        return name,pred_seqs,entropy

    for step, batch in enumerate(loader):
        err, count, hypref = eval_once(task,batch, encoder, decoder, device, opts.beam_size, opts.max_seq_len, opts.alpha)
        total_err += err
        total_count += count
        hyp_list.extend(hypref[0])
        ref_list.extend(hypref[1])

    wer = total_err/total_count
    rouge, bleu = eval_tool.eval_batch(ref_list, hyp_list)

    logging.info('----------{:s}------------'.format(task))
    logging.info('total count: {:d}'.format(total_count))
    logging.info('WER: {:f}'.format(100*wer))
    logging.info('rouge_L1_F: {:f}'.format(rouge))
    logging.info('bleu-1: {:f}'.format(bleu[0]))
    logging.info('bleu-2: {:f}'.format(bleu[1]))
    logging.info('bleu-3: {:f}'.format(bleu[2]))
    logging.info('bleu-4: {:f}'.format(bleu[3]))

    return rouge, bleu


def train_iter(opts):
    add=[]
    with open('./../../dataset/phoenix/rule_pseudo.txt') as f:
        lines = f.readlines()
        
    for x in tqdm(lines):
        x = x.strip().split('|')
        line = {
            'name': x[0],
            'label_word': x[1].split(' ')[:-1],
        }
        add.append(line)
        
    rule_add_gloss = {}
    for i,sample in enumerate(add):
        name=sample['name']
        rule_gloss=sample['label_word']
        rule_add_gloss[name]=[rule_gloss,1]
    gloss_pth=os.path.join(opts.log_dir,'gloss.pkl')
    with open(gloss_pth, 'wb') as f:
        pickle.dump(rule_add_gloss, f)
        
    best_model_pth=''
    T=1
    info_kit = InfoToolKit(gloss_pth,T,opts.tag, src='word', trg='gloss', external_corpus=opts.external_corpus, load_feature=True)
    logging.info(info_kit)
    info = info_kit.info
    num_glosses = info_kit.num_glosses
    num_words = info_kit.num_words

    global _BOS
    global _EOS
    global _BLK
    global _UNK
    _BOS = info_kit._trg_BOS
    _EOS = info_kit._trg_EOS
    _BLK = info_kit._trg_BLK
    _UNK = info_kit._trg_UNK

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = [
        FeatDataset(info[0], trg='gloss', external_corpus=info_kit.info_external_corpus,random_throw=True),
        FeatDataset(info[1], trg='gloss'),
        FeatDataset(info[2], trg='gloss'),
    ]
    collate = CollateFeat2Text(src_BLK=info_kit._src_BLK, tgt_BLK=_BLK, tgt_EOS=_EOS, pad_label=True)
    loader = [
        DataLoader(dataset[0], batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, collate_fn=collate.collate_fn_feat),
        DataLoader(dataset[1], batch_size=opts.test_batch_size, shuffle=False, num_workers=opts.num_workers, collate_fn=collate.collate_fn_feat),
        DataLoader(dataset[2], batch_size=opts.test_batch_size, shuffle=False, num_workers=opts.num_workers, collate_fn=collate.collate_fn_feat),
    ]

    encoder = EncoderTransformer(
            input_size=opts.emb_size, vocab_size=num_words, emb_dropout=opts.emb_dropout, hidden_size=opts.hidden_size,
            num_layers=opts.num_layers, num_heads=opts.num_heads, ff_size=opts.ff_size, dropout=opts.dropout, num_classes=-1
    )
    decoder = DecoderTransformer(
        vocab_size=num_glosses, emb_size=opts.emb_size, num_layers=opts.num_layers, num_heads=opts.num_heads,
        hidden_size=opts.hidden_size, ff_size=opts.ff_size, dropout=opts.dropout, emb_dropout=opts.emb_dropout
    )
    logging.info(encoder)
    logging.info(decoder)

    if opts.checkpoint_path != '':
        state_dict = torch.load(opts.checkpoint_path)
        encoder.load_state_dict(state_dict['encoder'])
        decoder.load_state_dict(state_dict['decoder'])
    encoder.to(device)
    decoder.to(device)

    encoder_optim = optim.Adam(
        [p for p in encoder.parameters() if p.requires_grad]+[p for p in decoder.parameters() if p.requires_grad], lr=opts.learning_rate, weight_decay=opts.weight_decay
    )
    decoder_optim = None#optim.Adam([p for p in decoder.parameters() if p.requires_grad], lr=opts.learning_rate, weight_decay=opts.weight_decay)
    #scheduler = StepLR(optimizer, step_size=opts.decay_epoch, gamma=opts.decay_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optim, mode='max', factor=0.5, patience=7, threshold_mode='abs', verbose=True)
    criterion = MaskedCrossEntropyLoss(pad_index=_BLK, smoothing=opts.smoothing)
    eval_tool = SLT_Eval(vocab=[chr(x) for x in range(20000, 20000+5000)])

    interval = opts.log_interval
    loss_am = AverageMeter()
    num_correct_am = AverageMeter()
    num_words_am = AverageMeter()
    best_dev = []
    best_epoch = 0
    model_list = []
    step = 0

    if opts.checkpoint_path != '':
        dev_rouge, dev_bleu = eval_iter(opts, 'dev', loader[1], encoder, decoder, device, eval_tool)
        dev_rouge, dev_bleu = eval_iter(opts, 'test', loader[2], encoder, decoder, device, eval_tool)

    for T in range(1,5):
        logging.info(T)
        info_kit = InfoToolKit(gloss_pth,T,opts.tag, src='word', trg='gloss', external_corpus=opts.external_corpus, load_feature=True)
        logging.info(info_kit)
        info = info_kit.info
        num_glosses = info_kit.num_glosses
        num_words = info_kit.num_words

        _BOS = info_kit._trg_BOS
        _EOS = info_kit._trg_EOS
        _BLK = info_kit._trg_BLK
        _UNK = info_kit._trg_UNK

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = [
            FeatDataset(info[0], trg='gloss', external_corpus=info_kit.info_external_corpus,random_throw=True),
            FeatDataset(info[1], trg='gloss'),
            FeatDataset(info[2], trg='gloss'),
        ]
        collate = CollateFeat2Text(src_BLK=info_kit._src_BLK, tgt_BLK=_BLK, tgt_EOS=_EOS, pad_label=True)
        loader = [
            DataLoader(dataset[0], batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, collate_fn=collate.collate_fn_feat),
            DataLoader(dataset[1], batch_size=opts.test_batch_size, shuffle=False, num_workers=opts.num_workers, collate_fn=collate.collate_fn_feat),
            DataLoader(dataset[2], batch_size=opts.test_batch_size, shuffle=False, num_workers=opts.num_workers, collate_fn=collate.collate_fn_feat),
        ]
        encoder = EncoderTransformer(
            input_size=opts.emb_size, vocab_size=num_words, emb_dropout=opts.emb_dropout, hidden_size=opts.hidden_size,
            num_layers=opts.num_layers, num_heads=opts.num_heads, ff_size=opts.ff_size, dropout=opts.dropout, num_classes=-1
        )
        decoder = DecoderTransformer(
            vocab_size=num_glosses, emb_size=opts.emb_size, num_layers=opts.num_layers, num_heads=opts.num_heads,
            hidden_size=opts.hidden_size, ff_size=opts.ff_size, dropout=opts.dropout, emb_dropout=opts.emb_dropout
        )
        encoder.to(device)
        decoder.to(device)

        encoder_optim = optim.Adam(
            [p for p in encoder.parameters() if p.requires_grad]+[p for p in decoder.parameters() if p.requires_grad], lr=opts.learning_rate, weight_decay=opts.weight_decay
        )
        for epoch in range(opts.global_epoch+T*10):
            logging.info('epoch: {:d}'.format(epoch))
            for i, param_group in enumerate(encoder_optim.param_groups):
                current_lr = float(param_group['lr'])
                logging.info('param group: {:d}, lr: {:.4e}'.format(i, current_lr))
            if opts.external_corpus:
                if epoch < (5+T*10):
                    dataset[0].change_mode('et')
                else:
                    dataset[0].change_mode('t')
            logging.info('num_traning_samples: {:d}'.format(len(loader[0].dataset)))
            half_step = len(loader[0].dataset) // opts.batch_size // 2
            logging.info('half_step: {:d}'.format(half_step))
            for _, batch in enumerate(loader[0]):
                loss, _, num_correct, num_words = train_once(epoch,
                    opts, batch,encoder, decoder, encoder_optim, decoder_optim, device, criterion)
                loss_am.update(loss)
                num_correct_am.update(num_correct)
                num_words_am.update(num_words)
                if step % interval == 0:
                    logging.info('epoch:{:d}, step:{:d}, loss:{:.3f}, acc:{:.2f}({:d}/{:d})'.format(
                        epoch, step, loss_am.avg, num_correct_am.sum/num_words_am.sum, num_correct_am.sum, num_words_am.sum))
                    loss_am.reset()
                    num_correct_am.reset()
                    num_words_am.reset()

                if step % half_step == 0 and step != 0:
                    logging.info('step_evaluation: {:d}'.format(step))
                    dev_rouge, dev_bleu= eval_iter(opts, 'dev', loader[1],encoder, decoder, device, eval_tool)
                    test_rouge, test_bleu = eval_iter(opts, 'test', loader[2],encoder, decoder, device, eval_tool)
                    
                    dev_sum = [dev_rouge, dev_bleu[-1]]

                    if sum(dev_sum) > sum(best_dev):
                        best_dev = dev_sum
                        best_epoch = epoch
                    
                    encoder.eval()
                    decoder.eval()
                    model_dict = {
                        'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                    }
                    model_name = [
                        sum(dev_sum), 
                        os.path.join(
                            opts.log_dir, 'v_r{:05.2f}_b{:05.2f}_t_r{:05.2f}_b{:05.2f}_s{:d}.pth'.format(dev_rouge, dev_bleu[-1], test_rouge, test_bleu[-1], step))
                    ]
                    torch.save(model_dict, model_name[1])
                    model_list.append(model_name)

                    for i, model_name in enumerate(sorted(model_list, key=lambda x: x[0], reverse=True)):
                        if i >= 5:
                            os.remove(model_name[1])
                            model_list.pop(model_list.index(model_name))
                    model_list=sorted(model_list, key=lambda x: x[0], reverse=True)
                    best_model_pth=model_list[0][1]

                step += 1
                
        with open('./../../dataset/phoenix/rule_pseudo.txt', 'r') as f:
            lines = f.readlines()

        add_gloss={}
        for x in lines:
            x = x.strip().split('|')
            add_gloss[x[0]]=[[['0'],1],[[0],0]]
        
        gloss_pth=os.path.join(opts.log_dir,'gloss.pkl')
        with open(gloss_pth, 'wb') as f:
            pickle.dump(add_gloss, f)

        info_kit = InfoToolKit(gloss_pth,2,opts.tag, src='word', trg='gloss', external_corpus=opts.external_corpus, load_feature=True)
        logging.info(info_kit)
        info = info_kit.info
        num_glosses = info_kit.num_glosses
        num_words = info_kit.num_words

        _BOS = info_kit._trg_BOS
        _EOS = info_kit._trg_EOS
        _BLK = info_kit._trg_BLK
        _UNK = info_kit._trg_UNK

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = [
            FeatDataset(info[0], trg='gloss', external_corpus=info_kit.info_external_corpus),
            FeatDataset(info[1], trg='gloss'),
            FeatDataset(info[2], trg='gloss'),
        ]
        collate = CollateFeat2Text(src_BLK=info_kit._src_BLK, tgt_BLK=_BLK, tgt_EOS=_EOS, pad_label=True)
        loader = [
            DataLoader(dataset[0], batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers, collate_fn=collate.collate_fn_feat),
            DataLoader(dataset[1], batch_size=opts.test_batch_size, shuffle=False, num_workers=opts.num_workers, collate_fn=collate.collate_fn_feat),
            DataLoader(dataset[2], batch_size=opts.test_batch_size, shuffle=False, num_workers=opts.num_workers, collate_fn=collate.collate_fn_feat),
        ]

        logging.info('best_sum: {}, epoch: {:d}'.format(best_dev, best_epoch))
        logging.info('epoch: {:d}'.format(epoch))
        for i, param_group in enumerate(encoder_optim.param_groups):
            current_lr = float(param_group['lr'])
            logging.info('param group: {:d}, lr: {:.4e}'.format(i, current_lr))
        dataset[0].change_mode('e')
        logging.info('num_traning_samples: {:d}'.format(len(loader[0].dataset)))
        half_step = len(loader[0].dataset) // opts.batch_size // 2
        logging.info('half_step: {:d}'.format(half_step))

        print(best_model_pth)
        state_dict = torch.load(best_model_pth)
        encoder.load_state_dict(state_dict['encoder'])
        decoder.load_state_dict(state_dict['decoder'])

        name,pred_seqs,entropy= eval_iter(opts, 'add', loader[0],encoder, decoder, device, eval_tool)
        entropy=list(entropy)

        add_gloss={}
        for i in range(len(name)):
            namei=name[i]
            glossi=pred_seqs[i]
            add_gloss[namei]=[rule_add_gloss[namei],[glossi,entropy[i]]]

        with open(gloss_pth, 'wb') as f:
            pickle.dump(add_gloss, f)



def parse_args():
    p = argparse.ArgumentParser(description='slr')
    p.add_argument('-cfg', '--config_file', type=str, default='config/translation/S2T-transformer.yaml')
    p.add_argument('-g', '--gpu', type=str, default='')
    p.add_argument('-t', '--tag', type=str)
    p.add_argument('-ds', '--dataset', type=str)
    p.add_argument('-ec', '--external_corpus', type=str, default='')

    p.add_argument('-bs', '--batch_size', type=int)
    p.add_argument('-tbs', '--test_batch_size', type=int)
    p.add_argument('-lr', '--learning_rate', type=float)
    p.add_argument('-wd', '--weight_decay', type=float)

    p.add_argument('-ge', '--global_epoch', type=int)
    p.add_argument('-dr', '--decay_rate', type=float)
    p.add_argument('-de', '--decay_epoch', type=int)

    p.add_argument('-m', '--model', type=str)
    p.add_argument('-rt', '--rnn_type', type=str)
    p.add_argument('-am', '--attn_method', type=str)
    p.add_argument('-is', '--input_size', type=int)
    p.add_argument('-es', '--emb_size', type=int)
    p.add_argument('-hs', '--hidden_size', type=int)
    p.add_argument('-fs', '--ff_size', type=int)
    p.add_argument('-nl', '--num_layers', type=int)
    p.add_argument('-nh', '--num_heads', type=int)
    p.add_argument('-ed', '--emb_dropout', type=float)
    p.add_argument('-d', '--dropout', type=float)
    p.add_argument('-if', '--input_feeding', action='store_true')
    p.add_argument('-sm', '--smoothing', type=float)

    p.add_argument('-p', '--pretrained', action='store_true', default=False)
    p.add_argument('-cp', '--checkpoint_path', type=str)

    p.add_argument('-bm', '--beam_size', type=int)
    p.add_argument('-lpa', '--alpha', type=float)
    p.add_argument('-msl', '--max_seq_len', type=int)
    p.add_argument('-tf', '--teacher_forcing', type=float)
    p.add_argument('-sp', '--sample', type=int)

    p.add_argument('-cd', '--change_dir', type=str, help='change current directory to save checkpoints at somewhere else')
    p.add_argument('-ld', '--log_dir', type=str, help='Leave it blank usually. To specify position for saving')
    p.add_argument('-lt', '--log_tail', type=str, help='add reference info at the end of folder name')
    p.add_argument('-li', '--log_interval', type=int, help=' interval of steps for logging training outputs')
    p.add_argument('-nw', '--num_workers', type=int, help='number of workers for dataloader')
    args = p.parse_args()
    
    cfg = load_config(args.config_file, with_prefix=True)
    for key, value in vars(args).items():
        if value is None:
            setattr(args, key, cfg[key])
    args.pytorch_version = torch.__version__

    return  args


import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    set_seed(seed=0)
    opts = parse_args()

    if opts.change_dir != '':
        os.chdir(opts.change_dir)

    if opts.gpu !=  '':
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    log_dir = os.path.join('./translation/seq_model', opts.dataset+'_'+time.strftime("%y%m%d%H%M%S", time.localtime()))
    log_dir = os.path.join(opts.log_dir, log_dir+opts.log_tail)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    init_logging(os.path.join(log_dir, 'log.txt'))
    opts.log_dir = log_dir

    logging.info(' '.join(sys.argv))
    items = list(vars(opts).items())
    items.sort()
    for key, value in items:
        logging.info('{:s}: {:}'.format(key, value))

    train_iter(opts)

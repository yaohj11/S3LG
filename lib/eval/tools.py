from itertools import groupby


def merge_same(seq_list):
    '''
        merge all same words in a list-format sentence.
    '''
    return  [x[0] for x in groupby(seq_list)]


def process_seq_for_slt(raw_seq, end_token, mode='first'):
    '''
        cut seq from end_token.
    '''
    x = [x[0] for x in groupby(raw_seq)]
    if end_token not in x:
        return x
    elif mode == 'first':
        pos = x.index(end_token)
        return x[:pos]
    else:
        pos = len(x)-1 - x[::-1].index(end_token)
        return x[:pos]

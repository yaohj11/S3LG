import spacy
import os
from tqdm import tqdm


with open('./phoenix2014T/text/gloss_map.txt', 'r') as f:
    gloss_map = [x.strip().lower() for x in f]
        
compound_gloss={}
for gloss in gloss_map:
    if '-' in gloss:
        for sub_word in gloss.split('-'):
            compound_gloss[sub_word]=gloss
                
nlp = spacy.load('de_core_news_lg')

add=[]
with open('./external_corpus.txt') as f:
    lines = f.readlines()
    
for x in tqdm(lines):
    x = x.strip().split('|')
    line = {
        'name': x[0],
        'label_word': x[2],
    }
    add.append(line)
    
rule_add_gloss = {}
for i,sample in enumerate(add):
    name=sample['name']
    text =sample['label_word']
    doc = nlp(text)
    rule_gloss=''
    for token in doc:
        if token.pos_=='PUNCT':
            continue
        token=token.lemma_.lower()
        if token in gloss_map:
            rule_gloss=rule_gloss+token+' '
        elif '-' in token:
            for sub_word in token.split('-'):
                if sub_word in gloss_map:
                    rule_gloss=rule_gloss+token+' '
                elif sub_word in list(compound_gloss.keys()):
                    gloss_or=compound_gloss[sub_word]
                    rule_gloss=rule_gloss+gloss_or+' '
                else:
                    rule_gloss=rule_gloss+sub_word+' '
    data=name+'|'+rule_gloss+'|'+text+'\n'
    with open('./rule_pseudo1.txt', 'a') as f:
        f.write(data)
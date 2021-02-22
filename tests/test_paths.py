from bs4 import BeautifulSoup
import pytest
import sys
sys.path.append("/Users/luidu652/Documents/causality_extraction/")
from dependency_paths import *


def test_paths():
    with open('pos_filtered_out_target_only_combined_topics.html') as ifile:
        samples = BeautifulSoup(ifile.read(), parser='html.parser')('p')
    gold_out = [(2, 'arbetslöshet//NN|UTR|SIN|IND|NOM//conj//0',
                 'medfört//VB|SUP|AKT//ROOT//8'),
                (1, 'arbetsmarknad//NN|UTR|SIN|IND|NOM//nsubj//8',
                 'medfört//VB|SUP|AKT//ROOT//8'),
                (2, 'arbetslöses//VB|PRS|SFO//nmod//27',
                 'medföra//VB|INF|AKT//conj//4'),
                (1, 'arbetslöshet//NN|UTR|SIN|IND|NOM//nsubj//5',
                 'medföra//VB|INF|AKT//ROOT//5'),
                (2, 'arbetslöshet//NN|UTR|SIN|IND|NOM//nmod//25',
                 'på//PP//case//25'),
                (1, 'arbetslöshet//NN|UTR|SIN|IND|NOM//obl//5',
                 'på//PP//case//9'),
                (3, 'arbetslöshet//NN|UTR|SIN|IND|NOM//nmod//8',
                 'medför//VB|PRS|AKT//ROOT//11'),
                (5, 'arbetslöshet//NN|UTR|SIN|IND|NOM//conj//6',
                 'medföra//VB|INF|AKT//acl:relcl//15'),
                (3, 'arbetslös//JJ|POS|UTR|SIN|IND|NOM//nsubj//9',
                 'på//PP//mark//26'),
                (3, 'arbetslös//JJ|POS|UTR|SIN|IND|NOM//nsubj//9',
                 'på//PP//mark//26')]

    for i in range(10):
        doc = model(samples[i].b.get_text(strip=True, separator=' '))
        parsed_text = ' '.join(['//'.join([tok.text, tok.tag_, tok.dep_,
                                           str(tok.head.i)]) for tok in doc])
        shortest, tokens = compute_path_length(str(samples[i]), parsed_text)
        assert gold_out[i] == (shortest, tokens[0], tokens[1]),\
            f'{gold_out[i]} != {(shortest, tokens[0], tokens[1])}'

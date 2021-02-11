from bs4 import BeautifulSoup
import pytest
import re, sys
sys.path.append("/Users/luidu652/Documents/causality_extraction/")
import os
from postprocessing import *

path = os.path.dirname(os.path.realpath(__file__))

def test_fix_file():
    test_file = f'{path}/test.html'
    file_content = '<html><head></head><body><p>Det här är en test</p></body></html>'
    with open(test_file, 'w') as ofile:
        ofile.write(file_content)
    fix_file(test_file)
    new_file = f'{path}/new_{test_file.split("/")[-1]}'
    with open(new_file) as ifile:
        file = re.sub('\n *', '', ifile.read())
    assert file == file_content.replace('ä', '&auml;'),\
        'character conversion changed since last update!'
    os.system(f'rm {test_file} {new_file}')


def test_remove_accent_chars():
    assert remove_accent_chars('éèêëåäáàãâíìï') == 'eeeeaaaaaaiii'


def test_separate_query_terms():
    query_exp = 'beror på | resultera | bidrar till'
    soup = BeautifulSoup('<em>Det <b>beror</b> ofta <b>på</b> att beteendet \
kan <b>resultera</b> i reaktioner som <b>bidrar</b> <b>till</b> negativa känslor.</em>',
                         features='lxml')
    terms = separate_query_terms(soup('b'), query_exp)
    assert terms == [[' *beror', '(ofta)? *på'],
                     ['(kan)? *resultera'],
                     ['(som)? *bidrar', ' *till']]


def test_redefine_boundaries():
    sents = ['Det här är en mening!',
            'Det gäller at övertyga EU:s befolkning.',
            '(1) Ibland har SpaCy problem med parenteser.']
    spacy_doc = model(' '.join(sents))
    new_boundaries = redefine_boundaries(spacy_doc)
    assert list(spacy_doc.sents) != sents
    assert new_boundaries == sents


def test_search():
    markup = '<match>Det skulle kunna motverka den stigande arbetslösheten.\
<b>Både sjukvårds- och miljöfrågor ...</b>\
Teknisk utveckling, konkurrens, förändrad efterfrågan och globalisering \
påverkar inte strukturomvandlingen direkt, utan via de företag som finns \
på marknaden.</match>'
    match = BeautifulSoup(markup, features='lxml').match
    topics = [['klimat', 'miljö'], ['vård'],
              ['tillväxt', 'utveckling'], ['arbetslöshet']]

    # target search
    target_search(match, topics)
    assert match['class'] == ['klimat', 'vård']

    # context search
    match = BeautifulSoup(markup, features='lxml').match
    context_search(match, topics)
    assert match['class'] == ['klimat', 'vård', 'tillväxt', 'arbetslöshet']


def test_format_txt_match():
    sents = ['Första meningen.',
             'Det här är en mening!',
            'Det gäller att övertyga EU:s befolkning.',
            '(1) Ibland har SpaCy problem med parenteser.']
    match = format_txt_match(sents, [2], context=2, highlight_query=False)
    formatted_match = [{'left': ['Första meningen.', 'Det här är en mening!'],
                        'right': ['(1) Ibland har SpaCy problem med parenteser.'],
                        'match': 'Det gäller att övertyga EU:s befolkning.'}]
    assert match == formatted_match


def test_xml_format_match():
    sents = ['Första meningen.',
             'Det här är en mening!',
            'Det gäller att övertyga EU:s befolkning.',
            '(1) Ibland har SpaCy problem med parenteser.']
    query_matches = [' *övertyga', 'befolkning *']
    match = str(format_xml_match(sents, [2], context=2, highlight_query=False)[0])
    xml_match = '<match>Första meningen. Det här är en mening!<em>Det gäller \
att övertyga EU:s befolkning.</em> (1) Ibland har SpaCy problem med parenteser.</match>'
    assert match == xml_match

    # with query term highlighting
    match = str(format_xml_match(sents, [2], context=2, highlight_query=True,
                             query_matches=query_matches)[0])
    xml_match = '<match>Första meningen. Det här är en mening!<em> Det gäller \
att <b>övertyga</b> EU:s <b>befolkning</b>.</em> (1) Ibland har SpaCy problem \
med parenteser.</match>'
    assert match == xml_match

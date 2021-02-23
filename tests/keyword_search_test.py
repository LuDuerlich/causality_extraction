import pytest
import re, sys
import os
current_path = os.path.realpath('__file__')
sys.path.append(current_path.strip('__file__'))
from whoosh import index, query, fields, analysis
from whoosh.util.testing import TempIndex
from whoosh.qparser import QueryParser
from whoosh.highlight import Highlighter
from keyword_search import *
from unused_code import BasicTokenizer, write_markup


text = """This is a sentence. Here's another one. How about some abbreviations, \
e.g. 'etc.' or 'et. al.'? How will this text end? We are about to find out. \
Maybe this can be the final sentence"""

def search(query_string, analyzer=None):
    schema = fields.Schema(doc_title=fields.TEXT(stored=True),
                           sec_title=fields.TEXT(stored=True),
                           body=fields.TEXT(analyzer, stored=True, phrase=True))
    sf = CustomSentenceFragmenter()
    formatter = CustomFormatter()
    highlighter = CustomHighlighter(fragmenter=sf, formatter=formatter, analyzer=analyzer)
    qp = QueryParser("body", schema=schema)

    results = []
    with TempIndex(schema) as ix:
        w = ix.writer()
        w.add_document(doc_title="doc1", sec_title="sec1",
                       body=text)
        w.commit()
        with ix.searcher() as s:
            q = qp.parse(query_string)
            result = s.search(q, terms=True)
            for hit in result:
                res = f"<match doc='{hit['doc_title']}' "
                res += f"section='{hit['sec_title']}'>"
                for s, _ in highlighter.highlight_hit(hit,"body", strict_phrase='"' in query_string):
                    res += f"<hit>{s}</hit>"
                res += "</match>"
                results.append(res)
    return results

def _test_query(terms, res, analyzer=analysis.StandardAnalyzer(stoplist=[])):
    for i, query in enumerate(terms):
        current_query = f"<query term='{query}'>"
        results = search(query, analyzer)
        for r in results:
            current_query += r
        current_query += "</query>"
        # print(current_query)
        # print(res[i]['full_match'])
        # check that text segments are highligted correctly
        assert res[i]["match_s"] in current_query
        # check that text segments are not duplicated or removed
        assert res[i]["full_match"] == current_query
    
def _single_term(analyzer=None):
    # StandardAnalyzer handles punctuation differently
    if not analyzer:
        analyzer = analysis.StandardAnalyzer(stoplist=[], minsize=1)
        terms = ['this', 'etc.']
        q1_match = "<query term='this'><match doc='doc1' section='sec1'><hit><b><em>This</em> is a sentence.</b> Here's another one.</hit><hit>or 'et. al.'? <b>How will <em>this</em> text end?</b> We are about to find out. Maybe this can be the final sentence</hit><hit>How will this text end? We are about to find out. <b>Maybe <em>this</em> can be the final sentence</b></hit></match></query>"
        q2_match = "<query term='etc.'><match doc='doc1' section='sec1'><hit>Here's another one. How about some abbreviations, e.g. '<b><em>etc</em>.</b>' or 'et.</hit></match></query>"
        q1_sent = "<b><em>This</em> is a sentence.</b>"
        q2_sent = "<b><em>etc</em>.</b>"

    else:
        terms = ['this', 'etc.']
        q1_match = "<query term='this'><match doc='doc1' section='sec1'><hit><b><em>This</em> is a sentence.</b> Here's another one.</hit><hit>. al.'<b>? How will <em>this</em> text end?</b> We are about to find out. Maybe this can be the final sentence</hit><hit>? How will this text end? We are about to find out<b>. Maybe <em>this</em> can be the final sentence</b></hit></match></query>"
        q2_match = "<query term='etc.'><match doc='doc1' section='sec1'><hit>. How about some abbreviations, e.g<b><em>.</em> '<em>etc.</em>' or 'et.</b> al.'?</hit><hit>. 'etc.' or 'et. al<b><em>.</em>'?</b> How will this text end?</hit><hit>? How will this text end? We are about to find out<b><em>.</em> Maybe this can be the final sentence</b></hit></match></query>"
        q1_sent = "<b><em>This</em> is a sentence.</b>"
        q2_sent = "<b><em>.</em> '<em>etc.</em>' or 'et.</b>"
    _test_query(terms, [{"match_s": q1_sent,
                         "full_match": q1_match},
                        {"match_s": q2_sent,
                         "full_match": q2_match}], analyzer)
        
def _multiple_terms(analyzer=None):
    if not analyzer:
        analyzer = analysis.StandardAnalyzer(stoplist=[], minsize=1)

        _test_query(['how about'],
                    [{"match_s":
                      "<b><em>How</em> <em>about</em> some abbreviations, e.g.</b>",
                      "full_match":
                      "<query term='how about'><match doc='doc1' section='sec1'><hit>This is a sentence. Here's another one. <b><em>How</em> <em>about</em> some abbreviations, e.g.</b> 'etc.</hit><hit>or 'et. al.'? <b><em>How</em> will this text end?</b> We are about to find out.</hit><hit>al.'? How will this text end? <b>We are <em>about</em> to find out.</b></hit></match></query>"}], analyzer)
    else:
        _test_query(['how about'],
                    [{"match_s":
                      "<b>. <em>How</em> <em>about</em> some abbreviations, e.</b>",
                      "full_match":
                      "<query term='how about'><match doc='doc1' section='sec1'><hit>This is a sentence. Here's another one<b>. <em>How</em> <em>about</em> some abbreviations, e.</b>g.</hit><hit>. al.'<b>? <em>How</em> will this text end?</b> We are about to find out.</hit><hit>.'? How will this text end<b>? We are <em>about</em> to find out.</b></hit></match></query>"}], analyzer)


def test_single_term():
    _single_term()


def test_multiple_terms():
    _multiple_terms()


def test_custom_tokeniser():
    analyzer = BasicTokenizer(do_lower_case=False) | analysis.LowercaseFilter()
    _single_term(analyzer)
    _multiple_terms(analyzer)


def test_write_markup():
    str_ = '<p>5 > 6 & "Hello World!" Hällö</p>'
    assert write_markup(str_, 'xml') == '<p>5 &gt; 6 &amp; "Hello World!" Hällö</p>'
    assert write_markup(str_, 'html') == '<p>5 &gt; 6 &amp; &quot;Hello World&excl;&quot; H&auml;ll&ouml;</p>'


def test_extract_sample():
    # since the data structure contains 210 samples,
    # we focus on keyword coverage instead of checking
    # the whole sample for equality
    sample = extract_sample()
    assert len(sample) == 21
    assert list(sample.keys()) == ['"bero på"', '"bidra till"',
                                   '"leda till"', '"på grund av"',
                                   '"till följd av"', 'följd',
                                   '"vara ett resultat av"', 'resultat',
                                   'resultera', 'därför', 'eftersom',
                                   'förklara', 'förorsaka', 'orsak',
                                   'orsaka', 'påverka', 'effekt',
                                   'medföra', 'framkalla', 'vålla',
                                   'rendera']
    assert len(sample['"bero på"']) == 10


def test_create_index():
    ix_path = '__test__index'
    if os.path.exists(ix_path):
        os.system(f'rm -r {ix_path}')
    filenames = ['documents/ft_H1B352.html']
    create_index(path_=ix_path, filenames=filenames, ixname=None)
    ix = index.open_dir(ix_path)
    assert ix.doc_count() == 23,\
        'Section segmentation is different from previous version!' +\
        f'should be 23 but the index now counts {ix.doc_count()}!'
    os.system(f'rm -r {ix_path}')


def test_format_simple_query():
    terms = ['medför//VB', 'tillväxt', 'växt//']
    expected_out = Or([Term('target', term) for term in terms])
    format_out = str(format_simple_query(terms))
    assert str(expected_out) == format_out,\
        f'abnormal output for simple query: {format_out}'


def test_format_parsed_query():
    field = 'parsed_target'
    terms = ['medför//VB', 'tillväxt', 'växt//']

    # regular
    expected_out = Or([And([Regex(field, terms[0]), Regex('target', terms[0])]),
                       And([Regex(field, terms[1]), Regex('target', terms[1])]),
                       And([Regex(field, terms[2]), Regex('target', terms[2])])])
    format_out = str(format_parsed_query(terms))
    assert str(expected_out) == format_out,\
        f'abnormal output for parsed query: {format_out}'
    
    # strict
    expected_out = Or([And([Regex(field, rf'^{terms[0]}'), Regex('target', rf'^{terms[0].split("//")[0]}')]),
                       And([Regex(field, rf'^{terms[1]}//'), Regex('target', rf'^{terms[1]}')]),
                       And([Regex(field, rf'^{terms[2]}'), Regex('target', rf'^{terms[2].split("//")[0]}')])])

    format_out = str(format_parsed_query(terms, strict=True))
    assert str(expected_out) == format_out,\
        f'abnormal output for strict parsed query: {format_out}'


def test_format_match():
    m = ('<b>some string with <em>some</em> html-style <em>highlighting</em></b>',
         {'doc_title': '<h1>Document 1</h1>', 'sec_title': 'The best section'})
    expected_out = "<match match_nb='2' doc='Document 1' section='The best section'><b>some string with <em>some</em> html-style <em>highlighting</em></b></match>"
    match = format_match(m, 2)
    assert expected_out == match,\
        f'incorrectly formatted match: {match}'

def test_redefine_boundaries():
    sents = ['Det här är en mening!',
            'Det gäller at övertyga EU:s befolkning.',
            '(1) Ibland har SpaCy problem med parenteser.']
    spacy_doc = model(' '.join(sents))
    new_boundaries, tokens = redefine_boundaries(spacy_doc)
    assert list(spacy_doc.sents) != sents
    assert new_boundaries == sents

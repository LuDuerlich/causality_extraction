import pytest
import re, sys
import os
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
        print(current_query)
        # check that text segments are highligted correctly
        assert res[i]["match_s"] in current_query
        # check that text segments are not duplicated or removed
        assert res[i]["full_match"] == current_query
    
def _single_term(analyzer=None):
    # StandardAnalyzer handles punctuation differently
    if not analyzer:
        analyzer = analysis.StandardAnalyzer(stoplist=[], minsize=1)
        terms = ['this', 'etc.']
        q1_match = "<query term='this'><match doc='doc1' section='sec1'><hit><em><b>This</b> is a sentence.</em> Here's another one.</hit><hit>or 'et. al.'? <em>How will <b>this</b> text end?</em> We are about to find out. Maybe this can be the final sentence</hit><hit>How will this text end? We are about to find out. <em>Maybe <b>this</b> can be the final sentence</em></hit></match></query>"
        q2_match = "<query term='etc.'><match doc='doc1' section='sec1'><hit>Here's another one. How about some abbreviations, e.g. '<em><b>etc</b>.</em>' or 'et.</hit></match></query>"
        q1_sent = "<em><b>This</b> is a sentence.</em>"
        q2_sent = "<em><b>etc</b>.</em>"

    else:
        terms = ['this', 'etc.']
        q1_match = "<query term='this'><match doc='doc1' section='sec1'><hit><em><b>This</b> is a sentence.</em> Here's another one.</hit><hit>. al.'<em>? How will <b>this</b> text end?</em> We are about to find out. Maybe this can be the final sentence</hit><hit>? How will this text end? We are about to find out<em>. Maybe <b>this</b> can be the final sentence</em></hit></match></query>"
        q2_match = "<query term='etc.'><match doc='doc1' section='sec1'><hit>. How about some abbreviations, e.g<em><b>.</b> '<b>etc.</b>' or 'et.</em> al.'?</hit><hit>. 'etc.' or 'et. al<em><b>.</b>'?</em> How will this text end?</hit><hit>? How will this text end? We are about to find out<em><b>.</b> Maybe this can be the final sentence</em></hit></match></query>"
        q1_sent = "<em><b>This</b> is a sentence.</em>"
        q2_sent = "<em><b>.</b> '<b>etc.</b>' or 'et.</em>"
    _test_query(terms, [{"match_s": q1_sent,
                         "full_match": q1_match},
                        {"match_s": q2_sent,
                         "full_match": q2_match}], analyzer)
        
def _multiple_terms(analyzer=None):
    if not analyzer:
        analyzer = analysis.StandardAnalyzer(stoplist=[], minsize=1)

        _test_query(['how about'],
                    [{"match_s":
                      "<em><b>How</b> <b>about</b> some abbreviations, e.g.</em>",
                      "full_match":
                      "<query term='how about'><match doc='doc1' section='sec1'><hit>This is a sentence. Here's another one. <em><b>How</b> <b>about</b> some abbreviations, e.g.</em> 'etc.</hit><hit>or 'et. al.'? <em><b>How</b> will this text end?</em> We are about to find out.</hit><hit>al.'? How will this text end? <em>We are <b>about</b> to find out.</em></hit></match></query>"}], analyzer)
    else:
        _test_query(['how about'],
                    [{"match_s":
                      "<em>. <b>How</b> <b>about</b> some abbreviations, e.</em>",
                      "full_match":
                      "<query term='how about'><match doc='doc1' section='sec1'><hit>This is a sentence. Here's another one<em>. <b>How</b> <b>about</b> some abbreviations, e.</em>g.</hit><hit>. al.'<em>? <b>How</b> will this text end?</em> We are about to find out.</hit><hit>.'? How will this text end<em>? We are <b>about</b> to find out.</em></hit></match></query>"}], analyzer)


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
    filenames = ['documents/ft_GTB366d4.html']
    create_index(path=ix_path, filenames=filenames, ixname=None)
    ix = index.open_dir(ix_path)
    assert ix.doc_count() == 127,\
        'Section segmentation not different from previous version!' +\
        f'should be 127 but the index now counts {ix.doc_count()}!'
    os.system(f'rm -r {ix_path}')

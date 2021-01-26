import pytest
import re, sys
from whoosh import index, query, fields, analysis
from whoosh.util.testing import TempIndex
from whoosh.qparser import QueryParser
from whoosh.highlight import Highlighter
from test import MyFormatter, MySentenceFragmenter, BasicTokenizer, MyHighlighter

text = """This is a sentence. Here's another one. How about some abbreviations, \
e.g. 'etc.' or 'et. al.'? How will this text end? We are about to find out. \
Maybe this can be the final sentence"""

def search(query_string, analyzer=None):
    schema = fields.Schema(doc_title=fields.TEXT(stored=True),
                           sec_title=fields.TEXT(stored=True),
                           body=fields.TEXT(analyzer, stored=True, phrase=True))
    sf = MySentenceFragmenter()
    formatter = MyFormatter()
    highlighter = MyHighlighter(fragmenter=sf, formatter=formatter, analyzer=analyzer)
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
                for s in highlighter.highlight_hit(hit,"body", strict_phrase='"' in query_string):
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
        # check that text segments are highligted correctly
        assert res[i]["match_s"] in current_query
        # check that text segments are not duplicated or removed
        assert res[i]["full_match"] == current_query
    
def _single_term(analyzer=None):
    # StandardAnalyzer handles punctuation differently
    if not analyzer:
        analyzer = analysis.StandardAnalyzer(stoplist=[], minsize=1)
        terms = ['this', 'etc.']
        q1_match = "<query term='this'><match doc='doc1' section='sec1'><hit><em><b>This</b> is a sentence.</em> Here's another one.</hit><hit>Here's another one. How about some abbreviations, e.g. 'etc.' or 'et. al.'? <em>How will <b>this</b> text end?</em> We are about to find out. Maybe this can be the final sentence</hit><hit>How will this text end? We are about to find out. <em>Maybe <b>this</b> can be the final sentence</em></hit></match></query>"
        q2_match = "<query term='etc.'><match doc='doc1' section='sec1'><hit>This is a sentence. Here's another one. <em>How about some abbreviations, e.g. '<b>etc</b>.' or 'et. al.</em>'? How will this text end?</hit></match></query>"
        q1_sent = "<em><b>This</b> is a sentence.</em>"
        q2_sent = "<em>How about some abbreviations, e.g. '<b>etc</b>.' or 'et. al.</em>"

    else:
        terms = ['this', 'etc.']
        q1_match = "<query term='this'><match doc='doc1' section='sec1'><hit><em><b>This</b> is a sentence. </em>Here's another one. </hit><hit>'etc.' or 'et. al.'? <em>How will <b>this</b> text end? </em>We are about to find out. Maybe this can be the final sentence</hit><hit>How will this text end? We are about to find out. <em>Maybe <b>this</b> can be the final sentence</em></hit></match></query>"
        q2_match = "<query term='etc.'><match doc='doc1' section='sec1'><hit>This is a sentence. Here's another one. <em>How about some abbreviations, e<b>.</b>g<b>.</b> '<b>etc.</b></em>'etc.' or 'et. al.'? How will this text end? </hit><hit>Here's another one. How about some abbreviations, e.g. <em>'etc.' or 'et<b>.</b> al<b>.</b>'? </em>How will this text end? We are about to find out. </hit><hit>How about some abbreviations, e.g. 'etc.' or 'et. al.'? <em>How will this text end? We are about to find out<b>.</b></em>We are about to find out. Maybe this can be the final sentence</hit></match></query>"
        q1_sent = "<em><b>This</b> is a sentence. </em>"
        q2_sent = "<em>How about some abbreviations, e<b>.</b>g<b>.</b> '<b>etc.</b></em>"
    _test_query(terms, [{"match_s": q1_sent,
                         "full_match": q1_match},
                        {"match_s": q2_sent,
                         "full_match": q2_match}], analyzer)
        
def _multiple_terms(analyzer=None):
    if not analyzer:
        analyzer = analysis.StandardAnalyzer(stoplist=[], minsize=1)

        _test_query(['how about'],
                    [{"match_s":
                      "<em><b>How</b> <b>about</b> some abbreviations, e.g. 'etc.' or 'et. al.</em>",
                      "full_match":
                      "<query term='how about'><match doc='doc1' section='sec1'><hit>This is a sentence. Here's another one. <em><b>How</b> <b>about</b> some abbreviations, e.g. 'etc.' or 'et. al.</em>'? How will this text end? We are about to find out.</hit><hit>Here's another one. How about some abbreviations, e.g. 'etc.' or 'et. al.'? <em><b>How</b> will this text end?</em> We are about to find out. Maybe this can be the final sentence</hit><hit>How about some abbreviations, e.g. 'etc.' or 'et. al.'? How will this text end? <em>We are <b>about</b> to find out.</em> Maybe this can be the final sentence</hit></match></query>"}], analyzer)
    else:
        _test_query(['how about'],
                    [{"match_s":
                      "<em><b>How</b> <b>about</b> some abbreviations, e.g. </em>",
                      "full_match":
                      "<query term='how about'><match doc='doc1' section='sec1'><hit>This is a sentence. Here's another one. <em><b>How</b> <b>about</b> some abbreviations, e.g. </em>'etc.'</hit><hit>'etc.' or 'et. al.'? <em><b>How</b> will this text end? </em>We are about to find out. Maybe this can be the final sentence</hit><hit>' or 'et. al.'? How will this text end? <em>We are <b>about</b> to find out. </em>Maybe this can be the final sentence</hit></match></query>"}], analyzer)

def test_single_term():
    _single_term()

def test_multiple_terms():
    _multiple_terms()

def test_custom_tokeniser():
    analyzer = BasicTokenizer(do_lower_case=False) | analysis.LowercaseFilter()
    _single_term(analyzer)
    _multiple_terms(analyzer)

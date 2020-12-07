import pytest
import re, sys
from whoosh import index, query, fields, analysis
from whoosh.util.testing import TempIndex
from whoosh.qparser import QueryParser
from test import MyFormatter, MySentenceFragmenter

text = """This is a sentence. Here's another one. How about some abbreviations, \
e.g. 'etc.' or 'et. al.'? How will this text end? We are about to find out. \
Maybe this can be the final sentence."""
analyzer = analysis.StandardAnalyzer(stoplist=[])
schema = fields.Schema(doc_title=fields.TEXT(stored=True),
                       sec_title=fields.TEXT(stored=True),
                       body=fields.TEXT(analyzer, stored=True, phrase=True))
sf = MySentenceFragmenter()
formatter = MyFormatter()
highlighter = Highlighter(fragmenter=sf, formatter=formatter)
qp = QueryParser("body", schema=schema)


def search(query_string):
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
                for s in highlighter.highlight_hit(hit,"body"):
                    res += f"<hit>{s}</hit>"
                res += "</match>"
                results.append(res)
    return results

def _test_query(terms, res):
    for i, query in enumerate(terms):
        current_query = f"<query term='{query}'>"
        results = search(query)
        for r in results:
            current_query += r
        current_query += "</query>"
        # check that text segments are highligted correctly
        assert res[i]["match_s"] in current_query
        # check that text segments are not duplicated or removed
        assert res[i]["full_match"] == current_query
    
def test_single_term():
    q1_sent = "<em><b>This</b> is a sentence.</em>"
    q1_match = "<query term='this'><match doc='doc1' section='sec1'><hit><em><b>This</b> is a sentence.</em> \
Here's another one.</hit><hit>or 'et. al.'? <em>How will <b>this</b> text end?</em> We are about to find out.\
 Maybe this can be the final sentence.</hit><hit>How will this text end? We are about to find out. <em>Maybe \
<b>this</b> can be the final sentence.</em></hit></match></query>"
    q2_sent = "<em><b>etc</b>.</em>"
    q2_match = "<query term='etc.'><match doc='doc1' section='sec1'><hit>Here's another one. How about some a\
bbreviations, e.g. '<em><b>etc</b>.</em>' or 'et.</hit></match></query>"
    _test_query(['this', 'etc.'],
            [{"match_s": q1_sent, "full_match": q1_match},
             {"match_s": q2_sent, "full_match": q2_match}])

def test_multiple_terms():
    _test_query(['"how about"'],
                [{"match_s":
                  "<em><b>How</b> <b>about</b> some abbreviations, e.g.</em>",
                  "full_match":
                  """<query term='"how about"'><match doc='doc1' section='sec1'><hit>This is a sentence. Here's another one. <em><b>How</b> <b>about</b> some abbreviations, e.g.</em> 'etc.</hit><hit>or 'et. al.'? <em><b>How</b> will this text end?</em> We are about to find out. Maybe this can be the final sentence.</hit><hit>al.'? How will this text end? <em>We are <b>about</b> to find out.</em> Maybe this can be the final sentence.</hit></match></query>"""}])

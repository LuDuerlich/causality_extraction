from data_extraction import Text
from bs4 import BeautifulSoup
import bs4.element
import copy
from custom_whoosh import *
from html.entities import html5
import logging
import os
from postprocessing import model, redefine_boundaries
import tarfile
import re
import sys
# sys.path.append("/Users/luidu652/Documents/causality_extraction/")
from search_terms import search_terms, expanded_dict,\
    incr_dict, decr_dict, annotated_search_terms,\
    keys_to_pos, filtered_expanded_dict,\
    create_tagged_term_list
# sys.path.append("/Users/luidu652/Documents/causality_extraction/whoosh/src/")
# from whoosh.analysis import SpaceSeparatedTokenizer
from util import find_nearest_neighbour
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED,\
    NUMERIC, FieldType
from whoosh.qparser import QueryParser, RegexPlugin
from whoosh.query import Regex, Or, And, Term, Phrase
# from whoosh.highlight import *
from whoosh import index, query, analysis
import random
import glob
import pickle


logging.basicConfig(filename="keyword_search.log",
                    filemode="w",
                    # level=logging.DEBUG
                    level=logging.INFO
                    )

path = os.path.dirname(os.path.realpath('__file__'))
analyzer = StandardAnalyzer(stoplist=[])
schema = Schema(doc_title=TEXT(stored=True, analyzer=analyzer,
                               lang='se'),
                sec_title=TEXT(stored=True, analyzer=analyzer,
                               lang='se'),
                target=TEXT(stored=True, phrase=True,
                            analyzer=analyzer,
                            lang='se'),
                parsed_target=TEXT(stored=True, phrase=True,
                            analyzer=analysis.SpaceSeparatedTokenizer() |\
                            analysis.LowercaseFilter(),
                            lang='se'),
                left_context=TEXT(stored=True, phrase=False,
                                  analyzer=analyzer, lang='se'),
                right_context=TEXT(stored=True, phrase=False,
                                   analyzer=analyzer, lang='se'),
                sent_nb=NUMERIC(stored=True, sortable=True))


def strip_tags(string):
    """strip title tags from string"""
    return re.sub("<h1>|</h1>", "", string)


def create_index(path_=f"{path}/test_index/", ixname="test", random_files=False,
                 schema=schema, add=False, filenames=None, n=5, parse=False):
    """Create or add to a specified index of files in documents.tar.

    Parameters:
               path (str):
                          path to index directory (should already exist)
                          (default='test_index'
               ixname (str):
                          name of the index to create / access
               random_files (bool):
                          whether or not to sample random files from
                          documents
               schema (Schema):
                          index schema specifying the fields within the
                          documents and how to index and access them
               add (bool):
                          whether or not to add the same previously
                          sampled files to the index
                          (useful to compare different index settings)
               n (int):
                          how many context sentences to store
                          (if None, stores all the context available
                          within the section)
    """

    if not path_.endswith("/"):
        path_ += "/"
    if os.path.exists(path_) and os.listdir(path_):
        if input(
                "index directory not empty. delete content?(Y/n) > "
        ).casefold() != "y":
            ix = index.open_dir(path_, indexname=ixname)
        else:
            logging.info(f'clearing out existing directory: {path_}')
            print(f"clearing out {path_} ...")
            os.system(f"rm -r {path_}*")
            ix = index.create_in(path_, schema, indexname=ixname)
    else:
        if not os.path.exists(path_):
            logging.info(f'creating new directory {path_}')
            print(f'creating directory {path_} ...')
            os.system(f'mkdir {path_}')
        ix = index.create_in(path_, schema, indexname=ixname)
    writer = ix.writer()
    if random_files:
        with open(f"{path}/ix_files.pickle", "rb") as ifile:
            seen = pickle.load(ifile)
            print(f'loading {len(seen)} seen files!')
        if not add:
            files = seen
        else:
            with tarfile.open(f"{path}/documents.tar", "r") as ifile:
                summaries = [fi for fi in ifile.getnames() if
                             fi.startswith('documents/s_')]
            files = random.sample([el for el in summaries
                                   if el not in seen], 500)
            # I don't remember why this is here
            seen.extend(files)
            with open(f"{path}/bt_ix_files.pickle", "wb") as ofile:
                pickle.dump(seen, ofile)
    elif filenames:
        print('index selected files')
        files = filenames
    else:
        files = ["H2B34", "H2B340", "H2B341", "H2B341", "H2B342",
                 "H2B343", "H2B344", "H2B345", "H2B346", "H2B347",
                 "H2B348", "H2B349", "H2B35"]
    logging.info(f'creating index: {ixname}')
    with tarfile.open(f"{path}/documents.tar", "r") as ifile:
        n_files = len(files)
        print(f'{n_files} to be indexed')
        logging.info(f'indexing {n_files} files')
        for i, key in enumerate(files):
            text = Text("")
            if key.endswith("html"):
                text.from_html(ifile.extractfile(key).read())
                key = key.split("_")[1].split(".")[0]
            else:
                text.from_html(
                    ifile.extractfile(
                        f"documents/s_{key}.html"
                    ).read())
            for j, section in enumerate(text):
                document = model(" ".join(section.text))
                sents = redefine_boundaries(document)
                for k, sent in enumerate(sents):
                    left_ctxt = ['']
                    right_ctxt = ['']
                    if k > 0:
                        if n is None:
                            left_ctxt = sents[:k]
                        else:
                            left_ctxt = sents[max(k-n, 0):k]
                    if k + 1 < len(sents):
                        if n is None:
                            right_ctxt = sents[k+1:]
                        else:
                            right_ctxt = sents[k+1:min(len(sents), k+1+n)]
                    target = str(sents[k])
                    title = re.sub(r'\s+', ' ', section.title)
                    # apparently, whoosh does not preserve newlines, so we have to
                    # mark sentence boundaries another way

                    right_ctxt = re.sub(r'\s+', ' ', "###".join(right_ctxt))
                    left_ctxt = re.sub(r'\s+', ' ', "###".join(left_ctxt))

                    if parse:
                        parsed_target = " ".join(['//'.join([token.text, token.tag_, token.dep_])
                                  for token in model(target)])
                        writer.add_document(doc_title=key,
                                            sec_title=title,
                                            left_context=left_ctxt,
                                            right_context=right_ctxt,
                                            target=target,
                                            parsed_target=parsed_target,
                                            sent_nb=k)

                    else:
                        writer.add_document(doc_title=key,
                                            sec_title=title,
                                            left_context=left_ctxt,
                                            right_context=right_ctxt,
                                            target=target,
                                            sent_nb=k)

            if i % 50 == 0:
                print(f'at file {i} ({text.title, k}), it has {j+1} sections')
                logging.info(f'at file {i} ({text.title, k}), it has {j+1} sections' +
                             f'the index currently contains {ix.doc_count()} documents.')
    writer.commit()


def print_to_file(keywords=["orsak", '"bidrar till"'], terms=[""], field=None):
    """Print all examples matching the  query term to an XML file.

    Parameters:
       keywords (list):
                        keywords to search for; here intended to indicate
                        causality (default=['orsak', '"bidrar till"'])
                        (note formatting for strict phrases with extra '"'
       terms (list):
                        additional terms to search for; e.g. 'klimatförendring'
                        (required: no)
       field (str):
                        the field to search if none is specified, 'target' or
                        'body' are searched depending on the schema
    """

    print('field:', field)
    if field is None:
        if 'target' in ix.schema._fields:
            field = 'target'
        elif 'body' in ix.schema._fields:
            field = 'body'
        else:
            raise Exception(f'Unknown schema without field specifier!' +
                            'If the index schema does not have a target' +
                            'or body field, specify another field to search!')
    print('field:', field)
    qp = QueryParser(field, schema=ix.schema)
    sentf = CustomSentenceFragmenter(maxchars=100000, context_size=4)
    formatter = CustomFormatter(between="\n...")
    highlighter = CustomHighlighter(fragmenter=sentf,
                                scorer=BasicFragmentScorer(),
                                formatter=formatter)

    punct = re.compile(r"[!.?]")
    filename = "example_queries_inflected.xml"
    print("Index:", ix, field)
    if terms[0]:
        filename = f"{terms[0]}_example_queries_inflected.xml"
    with ix.searcher() as s,\
         open(filename, "w") as output:
        soup = BeautifulSoup('', features='lxml')
        xml = soup.new_tag("xml")
        soup.append(xml)
        for term in terms:
            if term:
                _query = f"{term} AND ({'~2 OR '.join(keywords)})"
            else:
                _query = '~2 OR '.join(keywords)
            query = soup.new_tag('query', term=f'{_query}')
            xml.append(query)
            parsed_query = qp.parse(_query)
            r = s.search(parsed_query, terms=True, limit=None)
            if field == 'target':
                matches = []
                print('results:', len(r))
                for m in r:
                    hits = highlighter.highlight_hit(
                        m, field, top=len(m.results),
                        strict_phrase='"' in _query)
                    for hit, start_pos in hits:
                        matches.append((hit, m, m['sent_nb']))
            else:
                matches = [(hit, m, start_pos) for m in r
                           for hit in highlighter.highlight_hit(
                                   m, field, top=len(m.results),
                                   strict_phrase='"' in _query)]

            for i, matched_s in enumerate(matches):
                query.append(BeautifulSoup(format_match(matched_s[:-1], i), features='lxml'))
                # matched_s.results.order = FIRST
        output.write(soup.prettify())


def print_sample_file(keywords=expanded_dict, same_matches=None):
    """Sample 10 matches per query term and print them to an XML file.

    Parameters:
       keywords (dict):
                        dictionary of inflected terms organised by their
                        uninflected form (default=expanded_dict)
       same_matches (dict):
                        dictionary of uninflected terms, their correspond-
                        ing matches and metadata. Use only if same queries
                        should be sampled (default=None)
    """

    filename = "hit_sample.xml"
    if same_matches:
        with open(f"{path}/annotations/match_ids.pickle", "rb") as ifile:
            match_order = pickle.load(ifile)
        filename = filename.split(".")[0] + "_reconstructed.xml"
    total_matches = 0
    qp = QueryParser("body", schema=ix.schema)
    sentf = CustomSentenceFragmenter(maxchars=1000, context_size=4)
    formatter = CustomFormatter(between="\n...")
    highlighter = CustomHighlighter(fragmenter=sentf,
                              scorer=BasicFragmentScorer(),
                              formatter=formatter)

    punct = re.compile(r"[!.?]")
    xml = BeautifulSoup('<xml></xml>', features='lxml')
    with ix.searcher() as s:
        for key in list(expanded_dict):
            _query = '~2 OR '.join(expanded_dict[key])
            query = xml.new_tag('query', term=f'{key}({_query})')
            parsed_query = qp.parse(_query)
            r = s.search(parsed_query, terms=True, limit=None)
            matches = []
            nb_r = len(r)
            matches = [(hit, m, start_pos) for m in r
                       for hit, start_pos in highlighter.highlight_hit(
                               m, "body", top=len(m.results),
                               strict_phrase='"' in _query)]
            total_matches += len(matches)

            # limit to ten matches only
            print("nb of matches before sampling:", len(matches))
            if same_matches:
                match_ids = find_matched(matches,
                                         same_matches[key],
                                         match_order[key])
                for match, i in match_ids:
                    i = int(i)
                    if i < len(matches):
                        query.append(format_match(matches[i][:-1], nb, i))
            else:
                if len(matches) > 10:
                    match_ids = random.sample(matches, 10)
                else:
                    match_ids = list(range(len(matches)))
                for i, match in enumerate(match_ids):
                    query.append(format_match(match[:-1], i))

            print(f"Query {key}: {len(matches)} matches")
            xml.xml.append(query)
    print(f"{total_matches} total matches")
    with open(filename, "w") as output:
        output.write(xml.prettify())


def find_matched(matches, match_dict, order):
    """match hits to previously sampled matches by document, section ids
    and target sentence and order by original match number.

    This is necessary because the previously retrieved match numbers do not
    align in all cases. (May unfortunately also retrieve duplicate senteces
    within the same section)"""

    match_ids = []
    same_matched = copy.deepcopy(match_dict)
    for i, m in enumerate(matches):
        for match in same_matched:
            if m[1]["doc_title"] == match['doc'] and\
               m[1]['sec_title'] == match['st'] and\
               match['t'] in m[0]:
                match_ids.append([i, match['nb']])
                same_matched.remove(match)
        if len(match_ids) == 10:
            ordered = []
            print(len(match_ids), match_ids, order)
            for o in order:
                for mi in match_ids:
                    if str(o) == mi[1]:
                        ordered.append(mi)
            return ordered
    print(len(match_ids))
    ordered = []
    for o in order:
        for mi in match_ids:
            if str(o) == mi[1]:
                ordered.append(mi)

    return ordered


def extract_sample():
    """extract matches and metadata from previously sampled queries"""

    with open(f"{path}/samples/hit_sample.xml") as ifile:
        soup = BeautifulSoup(ifile.read(), features='lxml')
    queries = {}
    for query in soup.find_all("query"):
        matches = []
        for match in query.find_all("match"):
            matches.append({"nb": match['match_nb'],
                            "doc": match['doc'],
                            "st": match['section'],
                            "t": "".join([str(el) for el in
                                          match.em.contents])})
        queries[query['term'].split("(")[0]] = matches
    return queries


def format_match(match, match_nb, org_num=None, format_='xml'):
    if format_ == 'xml':
        tag = 'match'
    elif format_ == 'html':
        tag = 'p'

    title = match[1]['doc_title']
    sec_title = match[1]['sec_title']
    xml_match = f"<{tag} match_nb='{match_nb}"
    if org_num:
        xml_match += f"({org_num})' "
    else:
        xml_match += f"' "
    xml_match += (f"doc='{strip_tags(title)}' " +
                  f"section='{sec_title}'>")
    # remove keyword markup
    hit = re.sub(r'</?b>', '', match[0])
    # remove tags
    hit = ' '.join([token.split('//')[0] for token in hit.split()])
    # print(hit)
    xml_match += re.sub(r' ([?.,:;!])', r'\1', hit)
    xml_match += f"</{tag}>"
    return xml_match


def format_parsed_query(term_list, strict=False):
    if strict:
        return Or([Regex('parsed_target', fr"^{t}//") if '//' not in t
                   else Regex('parsed_target', fr"^{t}")
                   for t in term_list])
    return Or([Regex('parsed_target', fr"{t}") for t in term_list])


def format_simple_query(term_list):
    return Or([Term('target', term) for term in term_list])


def query_document(ix, id_="GIB33", keywords=['"bero på"', 'förorsaka'],
                   format_='xml', year='', additional_terms=[],
                   field=None, context_size=2, query_expansion=False,
                   exp_factor=3, prefix=""):
    """search for matches within one document
    Parameters:
               ix (FileIndex):
                  the index to search
               id_ (str):
                  the id of the document to search in the index
               keywords (list):
                  the (causality) query terms to match
               format (str):
                  the output format (either xml or html)
               year (str):
                  eventually the year to limit the search to
                  (for now just filename)
               additional_terms (list):
                  other terms to search in combination with the
                  keywords. To allow for multiple filters, this
                  is supposed to be a list of lists of words
               field (str):
                  the field to search; if the index schema has
                  a 'target' or 'body' field and no field is
                  prespecified, that fieldname is set.
               context_size (int):
                  the number of context sentences to the left and
                  right. Note that if the schema has designated fields
                  for left and right context, the context size
                  specified upon index creation is the upper limit.

    """
    sentf = CustomSentenceFragmenter(maxchars=1000,
                                    context_size=context_size)
    formatter = CustomFormatter(between="\n...")
    highlighter = CustomHighlighter(fragmenter=sentf,
                                scorer=BasicFragmentScorer(),
                                formatter=formatter)
    if field is None:
        if 'target' in ix.schema._fields:
            field = 'target'
        elif 'body' in ix.schema._fields:
            field = 'body'
        else:
            raise Exception(f'Unknown schema without field specifier!' +
                            'If the index schema does not have a target or' +
                            'body field, specify another field to search!')
    qp = QueryParser(field, schema=ix.schema)
    qp.add_plugin(RegexPlugin())
    text = Text('')
    text.from_html(f'documents/ft_{id_}.html')
    print(f'documents/ft_{id_}.html')
    matches_by_section = {}
    _query = ' OR '.join(keywords)
    prefix = f'{prefix}{id_}_{year}_'
    if query_expansion:
        prefix += 'extended_'
    if additional_terms:
        if 'parsed_target' in ix.schema.names():
            format_query = format_parsed_query
        else:
            format_query = format_simple_query
        for i, term_list in enumerate(additional_terms):
            if i > 0:
                if query_expansion:
                    term_list_ = []
                    for term in term_list:
                        term_list_.append(find_nearest_neighbour(term, exp_factor)[0])
                    additional_terms[i] = term_list_
                    terms = Or([format_query(q) for q in term_list_])
                else:
                    terms = format_query(term_list)
            else:
                terms = format_query(term_list, strict=True)
            _query = f"{terms} AND ({_query})"
    with ix.searcher() as searcher:
        query = qp.parse(
            f'((doc_title:{id_}) AND ({_query}))')
        # print(query)
        res = searcher.search(query, terms=True, limit=None)
        if field == 'target':
            matches = []
            print(len(res))
            for m in res:
                # print(len(m), len(res), m['sec_title'])
                hits = highlighter.highlight_hit(
                        m, "target", top=len(m.results),
                        strict_phrase='"' in _query)
                # print(len(hits))
                for hit, start_pos in hits:
                    matches.append((hit, m, m['sent_nb']))
        else:
            matches = [(hit, m, start_pos) for m in res
                       for hit, start_pos in highlighter.highlight_hit(
                               m, field, top=len(m.results),
                               strict_phrase='"' in _query)]

        for i, match in enumerate(matches):
            if match[1]['sec_title'] not in matches_by_section:
                matches_by_section[match[1]['sec_title']] = {}
            matches_by_section[match[1]['sec_title']][match[-1]] = \
                format_match(match[:-1], i, format_=format_)

    if format_ == 'xml':
        parser = 'lxml'
    else:
        parser = 'html.parser'
    output = BeautifulSoup(parser=parser)
    head = output.new_tag(format_)
    output.append(head)
    if format_ == 'html':
        style = output.new_tag('style')
        style.append("""
        body {
	margin-top: 4%;
	margin-bottom: 8%;
	margin-right: 13%;
	margin-left: 13%;
        }
        """)
        head.append(style)
    elif format_ == 'xml':
        query_el = output.new_tag('query')
        query_el['term'] = query
        head.append(query_el)
        head = query_el
    for section in text.section_titles:
        sec = output.new_tag('section')
        if format_ == 'html':
            heading = output.new_tag('h2')
            heading.append(section)
            sec.append(heading)
            head.append(sec)
            list_ = output.new_tag('ol')
            head.append(list_)
        else:
            heading = output.new_tag('title')
            heading.append(section)
            sec.append(heading)
        if section in matches_by_section:
            for key in sorted(matches_by_section[section]):
                el = BeautifulSoup(matches_by_section[section][key],
                                   parser=parser)
                if format_ == 'xml':
                    sec.append(el.match)
                else:
                    el.p.em.name = 'b'
                    for i, content in enumerate(el.p.b.contents):
                        if type(content) == bs4.element.Tag:
                            el.p.b.contents[i] = content.text

                    element = output.new_tag('li')
                    element.append(el.p)
                    list_.append(element)
        if format_ == 'xml':
            head.append(sec)
    if additional_terms:
        if len(additional_terms) > 1:
            with open(f'{path}/document_search/' +
                      f'{prefix}_incr_decr_custom' +
                      f'_document_search.{format_}',
                      'w') as ofile:
                if format_ == 'html':
                    ofile.write(output.prettify(formatter='html5'))
                else:
                    ofile.write(output.prettify())
        else:
            with open(f'{path}/document_search/' +
                      f'{prefix}_incr_decr_document_search.{format_}',
                      'w') as ofile:
                if format_ == 'html':
                    ofile.write(output.prettify(formatter='html5'))
                else:
                    ofile.write(output.prettify())
    else:
        with open(f'{path}/document_search/' +
                  f'{prefix}_document_search.{format_}',
                  'w') as ofile:
            if format_ == 'html':
                ofile.write(output.prettify(formatter='html5'))
            else:
                ofile.write(output.prettify())
    print(len(matches_by_section))
    return len(text.section_titles), len(matches)


if __name__ == "__main__":
    # analyzer = BasicTokenizer(do_lower_case=False) |\
    #    analysis.LowercaseFilter()
    analyzer = StandardAnalyzer(stoplist=[])
    schema = Schema(doc_title=TEXT(stored=True, analyzer=analyzer,
                                   lang='se'),
                    sec_title=TEXT(stored=True, analyzer=analyzer,
                                   lang='se'),
                    target=TEXT(stored=True, phrase=True,
                                analyzer=analyzer, lang='se'),
                    left_context=TEXT(stored=True, phrase=False,
                                      analyzer=analyzer, lang='se'),
                    right_context=TEXT(stored=True, phrase=False,
                                       analyzer=analyzer, lang='se'),
                    sent_nb=NUMERIC(stored=True, sortable=True))
    # ix = index.open_dir("yet_another_ix", indexname="big_index")
    # ix = index.open_dir("test_index", indexname="test")
    # ix = index.open_dir("bigger_index", indexname="big_index")
    # ix = index.open_dir("big_bt_index", indexname="bt_index")
    # query_list = [wf for term in expanded_dict.values() for wf in term]
    # print_to_file(query_list)
    # To create new index
    # create_index('bigger_index', schema=schema,
    # ixname='big_index', random_files=True)
    paths = ['documents/ft_GIB33.html', 'documents/ft_GKB3145d3.html',
             'documents/ft_GLB394.html', 'documents/ft_GOB345d1.html',
             'documents/ft_GRB350d3.html', 'documents/ft_GVB386.html',
             'documents/ft_GYB362.html', 'documents/ft_H1B314.html',
             'documents/ft_H4B391.html', 'documents/ft_H7B312.html']
    create_index('parsed_schema_document_ix', schema=schema, filenames=paths, parse=True)
    query_list = [wf for term in expanded_dict.values() for wf in term]
    decr_terms = [wf for term in {**decr_dict, **incr_dict}.values()
                  for wf in term]
    decr_terms_pos = [f'{wf}//{keys_to_pos[key]}' for key, term
                      in {**decr_dict, **incr_dict}.items()
                      for wf in term]
    ix = index.open_dir('parsed_schema_document_ix', indexname='test')
    years = ['1995', '1997', '1998', '2001', '2004', '2007', '2010',
             '2013', '2016', '2019']
    ids = ['GIB33', 'GKB3145d3', 'GLB394', 'GOB345d1', 'GRB350d3',
           'GVB386', 'GYB362', 'H1B314', 'H4B391', 'H7B312']
    topics = ['arbetslöshet', 'tillväxt', 'klimat', 'missbruk', 'hälsa']
    match_counter = 0
    sec_counter = 0
    format_ = 'html'
    # for i, id_ in enumerate(ids[1:2]):
    for i, id_ in enumerate(ids):
        print(years[i], id_)
        # secs, matches = query_document(ix, id_, keywords=query_list,
        #                                year=years[i], format_=format_,
        #                                field='target', context_size=2)

        # secs, matches = query_document(ix, id_, keywords=query_list,
        #                               year=years[i], format_=format_,
        #                               additional_terms=[decr_terms],
        #                               field='target')
        # secs, matches = query_document(ix, id_, keywords=query_list,
        #                                year=years[i], format_=format_,
        #                                additional_terms=[decr_terms] + [topics],
        #                                #[term for topic in topics for term in topic]],
        #                                field='target', query_expansion=True, exp_factor=10,
        #                                prefix='parsed_ix_')

        secs, matches = query_document(ix, id_, keywords=query_list,
                                       year=years[i], format_=format_,
                                       additional_terms=[decr_terms_pos],
                                       #[term for topic in topics for term in topic]],
                                       field='target',
                                       prefix='parsed_ix_pos_')

        print(secs, matches)
        sec_counter += secs
        match_counter += matches
    print(sec_counter, match_counter)


def find_causality_regions(filename):
    """situate sections with causality in a document"""
    with open(filename) as ifile:
        soup = BeautifulSoup(ifile.read())
    for i, section in enumerate(soup.find_all('section')):
        causal_sents = section.find_all('match')
        if causal_sents:
            print(f'{i}: {section.title.text.strip()} {len(causal_sents)}')
    print(len(soup.find_all('section')))

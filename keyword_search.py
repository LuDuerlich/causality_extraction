import datetime
from data_extraction import Text
from bs4 import BeautifulSoup
import copy
from custom_whoosh import CustomHighlighter, CustomSentenceFragmenter,\
    CustomFormatter, RegexPhrase
import logging
import os
import spacy
from spacy.tokens.span import Span
import tarfile
import traceback
import re
# import sys
# sys.path.append("/Users/luidu652/Documents/causality_extraction/")
from search_terms import search_terms, expanded_dict,\
    incr_dict, decr_dict, annotated_search_terms,\
    keys_to_pos, filtered_expanded_dict,\
    create_tagged_term_list
# sys.path.append("/Users/luidu652/Documents/causality_extraction/whoosh/src/")
# from whoosh.analysis import SpaceSeparatedTokenizer
from util import find_nearest_neighbour
from whoosh.fields import Schema, TEXT,\
    NUMERIC, DATETIME, ID
from whoosh.qparser import QueryParser, RegexPlugin
from whoosh.query import Regex, Or, And, Term, Phrase, SpanNear2
from whoosh.highlight import BasicFragmentScorer
from whoosh import index, analysis
import random
# import glob
import pickle
path = os.path.realpath('__file__').strip('__file__')
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logname = f'{datetime.datetime.now()}_keyword_search.log'


def setup_log(name, logname):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    filename = f"./test_{name}.log"
    log_handler = logging.FileHandler(filename)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(log_format)

    logger.addHandler(log_handler)

    return logger


logger = setup_log('whoosh_logger', logname)

model_path = f'{path}/sv_model_xpos/sv_model0/sv_model0-0.0.0/'
if not os.path.exists(model_path):
    model_path = f'{path}/spacy_model/sv_model_xpos/sv_model0/sv_model0-0.0.0/'
model = spacy.load(model_path)
sentencizer = model.create_pipe('sentencizer')
model.add_pipe(sentencizer)


with open('ids_to_date.pickle', 'rb') as ifile:
    ids_to_date = pickle.load(ifile)

path = os.path.dirname(os.path.realpath('__file__'))
analyzer = analysis.StandardAnalyzer(stoplist=[])
schema = Schema(id=ID(stored=True, unique=True),
                doc_title=TEXT(stored=True, analyzer=analyzer,
                               lang='se'),
                date=DATETIME(sortable=True),
                sec_title=TEXT(stored=True, analyzer=analyzer,
                               lang='se'),
                target=TEXT(stored=True, phrase=True,
                            analyzer=analyzer,
                            lang='se'),
                parsed_target=TEXT(stored=True, phrase=True,
                                   analyzer=analysis.SpaceSeparatedTokenizer()
                                   | analysis.LowercaseFilter(),
                                   lang='se'),
                left_context=TEXT(stored=True, phrase=False,
                                  analyzer=analyzer, lang='se'),
                right_context=TEXT(stored=True, phrase=False,
                                   analyzer=analyzer, lang='se'),
                sent_nb=NUMERIC(stored=True, sortable=True))


def redefine_boundaries(sents):
    """correct sentence boundaries of spacy sentencizer
    based on rules for abbreviation and possessive markers
    Parameters:
               sents (spacy.tokens.span.Span):
                           a spacy sents generator of Span objects
    """

    ents = [str(ent) for ent in sents.ents]
    sents = list(sents.sents)
    abr_exp = re.compile(r"(m\.m|osv|etc)\.")
    poss_exp = re.compile(r"\b[A-ZÄÖÅ0-9]+\b:$")
    for i in range(len(sents)):
        if i+1 >= len(sents):
            break
        has_abbrev = abr_exp.findall(str(sents[i]))[::-1]
        if has_abbrev:
            if type(sents[i]) == Span:
                tokens = list(sents[i].__iter__())
            else:
                tokens = sents[i].split()
            last = None
            while has_abbrev:
                nb_abbr = len(has_abbrev)
                for j, t in enumerate(tokens):
                    if not has_abbrev:
                        break
                    if has_abbrev[-1] in str(t):
                        if j+1 < len(tokens) and\
                           (str(tokens[j+1]).istitle() and
                            str(tokens[j+1]) not in ents):
                            has_abbrev.pop(-1)
                            new_s = " ".join(
                                [str(tok) for tok in tokens[j+1:]])
                            following = sents[i+1:]
                            sents[i] = " ".join(
                                [str(tok) for tok in tokens[:j+1]])
                            sents[i+1] = new_s
                            sents = sents[:i+2]
                            sents.extend(following)
                if nb_abbr == len(has_abbrev):
                    has_abbrev.pop(-1)

        # possessives of acronyms etc. tend to get split at the colon
        # i.e. 'EU:s direktiv ...' -> 'EU:', 's direktiv ...'
        has_poss = poss_exp.findall(str(sents[i]))
        split_on_poss = (has_poss and
                         (i + 1 < len(sents)
                          and re.match('[a-zäåö]', str(sents[i+1])[:2])))
        if split_on_poss:
            sents[i] = re.sub(r" ([.,;:!?])", r"\1",
                              str(sents[i]) + str(sents[i+1]))
            del sents[i+1]
        else:
            sents[i] = re.sub(r" ([.,;:!?])", r"\1", str(sents[i]))

        # sentences that start with parentheses are split at open parentheses
        if str(sents[i]).endswith("(") and i + 1 < len(sents):
            sents[i] = str(sents[i]).rstrip(' (')
            sents[i+1] = '(' + str(sents[i+1]).lstrip()
        sents = [str(s) for s in sents]
    return sents


def strip_tags(string):
    """strip title tags from string"""
    return re.sub("<h1>|</h1>", "", string)


def create_index(path_=f"{path}/test_index/", ixname="test",
                 random_files=False, schema=schema, add=False,
                 filenames=None, n=5, parse=False):
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
    print()
    print('PATH:', path_, os.path.exists(path_))
    print()
    if not path_.endswith("/"):
        path_ += "/"
    if os.path.exists(path_) and os.listdir(path_):
        if input(
                "index directory not empty. delete content?(Y/n) > "
        ).casefold() != "y":
            print(f'opening exisitin index {path_}')
            ix = index.open_dir(path_, indexname=ixname)
        else:
            logger.info(f'clearing out existing directory: {path_}')
            print(f"clearing out {path_} ...")
            os.system(f"rm -r {path_}*")
            ix = index.create_in(path_, schema, indexname=ixname)
    else:
        if not os.path.exists(path_):
            logger.info(f'creating new directory {path_}')
            print(f'creating directory {path_} ...')
            os.system(f'mkdir {path_}')
        ix = index.create_in(path_, schema, indexname=ixname)
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
    logger.info(f'creating index: {ixname}')
    mem = 2096
    writer = ix.writer(limitmb=mem/4)#, procs=4)
    with tarfile.open(f"{path}/documents.tar", "r") as ifile:
         #ix.writer(limitmb=mem/4, procs=4) as writer:
         # ix.writer(limitmb=mem/4, procs=4, multisegment=True) as writer:
        logger.info(f'writing with {mem/4} if memory on 4 processes ({mem})')
        n_files = len(files)
        print(f'{n_files} to be indexed')

        logger.info(f'indexing {n_files} files')

        try:
            for i, key in enumerate(files):
                text = Text("")
                if key.endswith("html"):
                    if os.path.exists(key):
                        text.from_html(key)
                    else:
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
                    # parsed_sents = model(sents)
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
                        # apparently, whoosh does not preserve newlines, so we have
                        # to mark sentence boundaries another way

                        right_ctxt = re.sub(r'\s+', ' ', "###".join(right_ctxt))
                        left_ctxt = re.sub(r'\s+', ' ', "###".join(left_ctxt))

                        # add date
                        if hasattr(datetime.datetime, 'fromisoformat'):
                            date = datetime.datetime.fromisoformat(
                                ids_to_date[key][0][1])
                        else:
                            year, month, day = ids_to_date[key][0][1].split('-')
                            date = datetime.datetime(int(year), int(month), int(day))

                        # To Do: parse all sentences in one go
                        if parse:
                            parsed_target = " ".join(['//'.join([token.text,
                                                                 token.tag_,
                                                                 token.dep_,
                                                                 str(token.head.i)])
                                                      for token in model(target)])
                            writer.add_document(id=".".join([str(nb) for nb in (key,j,k)]),
                                                doc_title=key,
                                                date=date,
                                                sec_title=title,
                                                left_context=left_ctxt,
                                                right_context=right_ctxt,
                                                target=target,
                                                parsed_target=parsed_target,
                                                sent_nb=k)

                        else:
                            writer.add_document(doc_title=key,
                                                date=date,
                                                sec_title=title,
                                                left_context=left_ctxt,
                                                right_context=right_ctxt,
                                                target=target,
                                                sent_nb=k)

                if i % 50 == 0:
                    print(f'{datetime.datetime.now()} at file {i} ({text.title, k}), it has {j+1} sections')
                    logging.info(f'at file {i} ({text.title, k}), it has {j+1} ' +
                                 'sections the index currently contains ' +
                                 f'{ix.doc_count()} documents.')
        except:

            traceback.print_exc()
            print(f'stopped at file {i} {key}, {j} {section}')
            writer.commit()
            return
        print('all done')
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
            raise Exception('Unknown schema without field specifier!' +
                            'If the index schema does not have a parsed_target' +
                            'or body field, specify another field to search!')
    print('field:', field)
    qp = QueryParser(field, schema=ix.schema)
    sentf = CustomSentenceFragmenter(maxchars=100000, context_size=4)
    formatter = CustomFormatter(between="\n...")
    highlighter = CustomHighlighter(fragmenter=sentf,
                                    scorer=BasicFragmentScorer(),
                                    formatter=formatter)

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
                matches = [(hit[0], m, hit[-1]) for m in r
                           for hit in highlighter.highlight_hit(
                                   m, field, top=len(m.results),
                                   strict_phrase='"' in _query)]

            for i, matched_s in enumerate(matches):
                query.append(BeautifulSoup(format_match(matched_s[:-1], i),
                                           features='lxml'))
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
                        query.append(format_match(matches[i][:-1], nb_r, i))
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
        xml_match += "' "
    xml_match += (f"doc='{strip_tags(title)}' " +
                  f"section='{sec_title}'>")
    # remove keyword markup
    # hit = re.sub(r'</?b>', '', match[0])
    hit = match[0]
    # remove tags
    hit = ' '.join([token.split('//')[0] for token in hit.split()])
    # print(hit)
    xml_match += re.sub(r' ([?.,:;!])', r'\1', hit)
    xml_match += f"</{tag}>"
    return xml_match


def format_parsed_query(term_list, strict=False):
    if strict:
        terms = []
        for term in term_list:
            if '//' in term:
                terms.append(And([Regex('parsed_target', fr"^{term}"),
                                  Regex('target', fr"^{term.split('//')[0]}")]
                                 ))
            else:
                terms.append(And([Regex('parsed_target', fr"^{term}//"),
                                  Regex('target', fr"^{term}")]))

        return Or(terms)
    return Or([And([Regex('parsed_target', term),
                    Regex('target', term)]) for term in term_list])


def format_simple_query(term_list):
    return Or([Term('target', term) for term in term_list])


def format_keyword_queries(keywords, field, qp, slop=1):
    terms = []
    for keyword in keywords:
        if '"' in keyword:
            terms.append(Phrase(field, keyword.strip('"').split(), slop=slop))
        elif '//' in keyword:
            terms.append(And([format_parsed_query([keyword], True)[0],
                              Term(field, keyword.split('//')[0])]))
        else:
            terms.append(qp.parse(f'{field}:{keyword}'))
    return Or(terms)


def query_document(ix, id_="GIB33", keywords=['"bero på"', 'förorsaka'],
                   format_='xml', year='', additional_terms=[],
                   field=None, context_size=2, query_expansion=False,
                   exp_factor=3, prefix="", slop=1):
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
               query_expansion (bool):
                  whether the additional terms should be expanded using
                  word embeddings
               exp_factor (int):
                  how many neighbours should be added during query
                  expansion
               prefix (str):
                  prefix for the output file
               slop (int):
                  slop factor for phrase queries (i.e. how many tokens
                  are allowed in between phrase constituents)

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
            raise Exception('Unknown schema without field specifier!' +
                            'If the index schema does not have a parsed_target or' +
                            'body field, specify another field to search!')
    qp = QueryParser(field, schema=ix.schema)
    qp.add_plugin(RegexPlugin())
    text = Text('')
    text.from_html(f'documents/ft_{id_}.html')
    print(f'documents/ft_{id_}.html')
    matches_by_section = {}
    _query = format_keyword_queries(keywords, field, qp, slop)
    prefix = f'{prefix}{id_}_{year}_'
    if query_expansion:
        prefix += 'extended_'
    if additional_terms:
        if 'target' in ix.schema.names():
            format_query = format_parsed_query
        else:
            format_query = format_simple_query
        for i, term_list in enumerate(additional_terms):
            if i > 0:
                if query_expansion:
                    term_list_ = []
                    for term in term_list:
                        term_list_.append([nn.replace('##', '\\B')
                                           for nn, sim in
                                           find_nearest_neighbour(term,
                                                                  exp_factor)])
                    additional_terms[i].append(term_list_)
                    terms = Or([format_query(q) for q in term_list_])
                else:
                    terms = format_query(term_list)
            else:
                terms = format_query(term_list, strict=True)
            if terms:
                _query = And([terms, _query])
    with ix.searcher() as searcher:
        query = And([qp.parse(f'doc_title:{id_}'), _query])
        logger.info(f'Query: {query}')
        res = searcher.search(query, terms=True, limit=None)
        if field == 'target':
            matches = []
            print(len(res))
            for m in res:
                hits = highlighter.highlight_hit(
                        m, field, top=len(m.results),
                        strict_phrase=True)
                for hit, start_pos in hits:
                    matches.append((hit, m, m['sent_nb']))
        else:
            matches = [(hit, m, start_pos) for m in res
                       for hit, start_pos in highlighter.highlight_hit(
                               m, field, top=len(m.results),
                               strict_phrase=True)]

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
    return len(text.section_titles), len(matches), additional_terms


if __name__ == "__main__":
    # analyzer = BasicTokenizer(do_lower_case=False) |\
    #    analysis.LowercaseFilter()
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
    create_index('parsed_schema_document_ix', schema=schema, filenames=paths,
                 parse=True)
    query_list = [wf for term in expanded_dict.values() for wf in term]
    filtered_query_list = [wf for term in filtered_expanded_dict.values()
                           for wf in term]
    tagged_list = create_tagged_term_list(filtered_expanded_dict,
                                          annotated_search_terms)
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
    topics = ['arbetslöshet', 'klimat', 'missbruk', 'hälsa']
    # topics = ['arbetslöshet', 'tillväxt', 'klimat', 'missbruk', 'hälsa']
    match_counter = 0
    sec_counter = 0
    format_ = 'html'
    for i, id_ in enumerate(ids[:1]):
        print(years[i], id_)
        # secs, matches = query_document(ix, id_, keywords=query_list,
        #                                year=years[i], format_=format_,
        #                                field='target', context_size=2)

        # secs, matches = query_document(ix, id_, keywords=query_list,
        #                               year=years[i], format_=format_,
        #                               additional_terms=[decr_terms],
        #                               field='target')
        secs, matches, expanded_terms = query_document(ix, id_,
                                                       keywords=tagged_list,
                                                       year=years[i],
                                                       format_=format_,
                                                       additional_terms=[[]] +
                                                       [list(topics)],
                                                       # [term for topic in
                                                       # topics for term in topic]],
                                                       field='target',
                                                       query_expansion=True,
                                                       exp_factor=20,
                                                       prefix='pos_filtered_topics_only_',
                                                       slop=1)

        # secs, matches, expanded_terms = query_document(ix, id_, keywords=query_list,
        #                                year=years[i], format_=format_,
        #                                additional_terms=[[]] + [list(topics)],
        #                                #[term for topic in topics for term in topic]],
        #                                field='target', query_expansion=True, exp_factor=20,
        #                                prefix='topics_only_')

        # secs, matches = query_document(ix, id_, keywords=query_list,
        #                                year=years[i], format_=format_,
        #                                additional_terms=[decr_terms_pos],
        #                                #[term for topic in topics for term in topic]],
        #                                field='target',
        #                                prefix='parsed_ix_pos_')

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

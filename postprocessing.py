from bs4 import BeautifulSoup
from spacy.tokens.span import Span
from search_terms import expanded_dict
import spacy
import pytest
import copy
import csv
import re
import regex
import unicodedata

def remove_accent_chars(x: str):
    return regex.sub(r'\p{Mn}', '', unicodedata.normalize('NFKD', x))

with open("samples/hit_samplereconstructed.xml") as ifile:
    mark_up = BeautifulSoup(ifile.read(), features='lxml')
# markup
matches = mark_up.find_all('match')
queries = mark_up.find_all('query')

# string representations
hits = [match.text for match in matches]
model_path = 'spacy_model/sv_model_xpos/sv_model0/sv_model0-0.0.0/'
model = spacy.load(model_path)
sentencizer = model.create_pipe('sentencizer')
model.add_pipe(sentencizer)


def try_language_models(m1=None, m2=None, m3=None, debug=False, ix=[0]):
    """compares two or more language models on the first
    match in hits. If no model names are given, default
    is to test English, Norwegian and Multilingual
    Parameters:
               m1, m2, m3 (str):
                         name of or (if needed) path to model
                         directory
               debug (bool):
                         shows sentences generated by each model
                         if set to True
               ix (iter):
                         indeces of the examples to look at in hits
    """

    models = []
    if not m1 and not m2:
        m1 = "en_core_web_sm"
        m2 = "nb_core_news_sm"
        m3 = "xx_ent_wiki_sm"

    for m in [m1, m2, m3]:
        if m:
            model = spacy.load(m)
            sentencizer = model.create_pipe('sentencizer')
            model.add_pipe(sentencizer)
            models.append(model)
    for i in ix:
        if debug:
            print(f'at hit {i+1}')
        example = hits[i].strip()
        docs = []
        for model in models:
            docs.append(model(example))
        if len(docs) > 2:
            last = docs[2]
        else:
            last = None
            compare_segmentation(docs[0], docs[1], last, debug)
    if debug:
        print("Done!")
    return True


def compare_segmentation(a, b, c=None, debug=False):
    """check if two or three sentencizer segmentations are the same
    shows the corresponding sentences one by one if debug is True.
    Parameters:
               a, b, c (spacy.tokens.doc.Doc):
                      a Doc object created by running a pipeline
                      containing a sentecizer on a text.
                      (c is optional)
               debug (bool):
                      mode that displays sentences side by side
                      for each Doc.
    """

    a_gen = a.sents
    b_gen = b.sents
    if c:
        c_gen = c.sents
    for a_s in a_gen:
        # compare sentences on token basis
        a_s = [tok.text for tok in a_s]
        for b_s in b_gen:
            b_s = [tok.text for tok in b_s]
            if c and c_gen:
                for c_s in c_gen:
                    c_s = [tok.text for tok in c_s]
                    if debug:
                        print(f" 1) {a_s}\n 2) {b_s}\n 3) {c_s}\n")
                    else:
                        assert a_s == b_s,\
                            f"Sentences do not match up:\n 1) {a_s}\n 2) {b_s}"
                        assert a_s == c_s,\
                            f"Sentences do not match up:\n 1) {a_s}\n 3) {c_s}"
                    break
            elif debug:
                print(f" 1) {a_s}\n 2) {b_s}\n")
            else:
                assert a_s == b_s,\
                    f"Sentences do not match up:\n 1) {a_s}\n 2) {b_s}"
            break
    return True


def test_lms():
    """compare the two Swedish language models"""

    # we are interested in the Swedish models
    m1 = 'spacy_model/sv_model_xpos/sv_model0/sv_model0-0.0.0/'
    m2 = 'spacy_model/sv_model_upos/sv_model0/sv_model_upos0-0.0.0/'

    # and potentially the multilingual one
    m3 = 'xx_ent_wiki_sm'
    assert try_language_models(m1, m2, ix=range(len(hits)))


def segment_match(match, query, highlight_query=False, context=2,
                  redo_boundaries=True, xml=True):
    """Split a whoosh match into sentences
    Parameters:
               match (bs4.element.Tag):
                     query match in context
               query (str):
                     representation of the original query
               highlight_query (bool):
                     whether or not the query term(s) should be
                     highlighted within the matched sentence
               context (int):
                     how many sentences of context to display around
                     the matched sentence
               redo_boundaries (bool):
                     whether or not the original sentence boundaries
                     should be corrected to account for frequent
                     abbreviation errors

    for now uses the Swedish xpos model
    """

    full_text = match.text
    query_matches = None
    # potentially useful later if the match sequence gets segmented
    # into multiple sentences
    q_ex = None
    match_sequence = None
    if match.b:
        query_matches = separate_query_terms(match("b"), query)
        query_matches = [r" *(\b.+\b *)?".join(t) for t in query_matches]
        q_ex = re.compile(r'|'.join(query_matches))
    if match.em:
        match_sequence = match.em.text
    else:
        print("no match?", match, query)
    sents = []
    match_id = []
    match_by_term = []
    match_first_term = []
    spacy_sents = model(full_text)
    sequence = list(spacy_sents.sents)
    if redo_boundaries:
        sequence = redefine_boundaries(spacy_sents)
    # locate the matched sentence
    j = 0
    for i, sent in enumerate(sequence):
        if sent:
            sents.append(sent)
            if match_sequence and match_sequence in str(sent):
                match_id.append(i)
            elif q_ex and q_ex.search(str(sent)):
                match_by_term.append(i)
            elif query_matches and query_matches[j] in str(sent):
                match_first_term.append(i)
                j += 1
    # fall-back if the match sequence is now segmented differently
    if not match_id and match_by_term:
        match_id = match_by_term
    if not match_id and match_first_term:
        match_id = match_first_term
    try:
        match_s = " ".join([str(sequence[j]) for j in match_id])
    except:
        print([sequence[j] for j in match_id])
        print(query_matches, q_ex, end=", ")
        print(match.text, match.em)
    assert q_ex.search(match_s),\
        f"Search term not in match ('{q_ex}', '{match_s}', " +\
        f"'{match_id}', {match.text}, {match.em})"
    if xml:
        return format_xml_match(sents, match_id,
                                context, highlight_query,
                                query_matches)
    else:
        return format_match(sents, match_id,
                            context, highlight_query)


def separate_query_terms(query_terms, query_exp):
    terms = []
    new_term = True
    for i, term in enumerate(query_terms):
        t = term.text.strip()
        context = ""
        # add previous token to the match string
        previous = model(str(term.previous_sibling))
        if str(previous) not in ["None", " ", ""]:
            previous = str(previous[-1])
            if i > 0 and previous not in [query_terms[i-1], ".", "?", "!"]:
                context = re.sub(r'([\[\])(\\?+*])', r'\\\1', previous)
                # print(f'previous: "{previous}" "{context}"')
        if not new_term:
            if f'| {" ".join(terms[-1])} |'.casefold()\
                                           .replace('*', '') in query_exp:
                new_term = True
            elif f'{" ".join(terms[-1] + [t])} '.casefold()\
                                                .replace('*', '') in query_exp:
                terms[-1].append(f'{context} *{t}')
                continue
        # start of a new term
        if new_term:
            if f'| {t} |'.casefold() in query_exp:
                terms.append([f'{context} *{t}'])
            elif f'| {t} '.casefold() in query_exp:
                terms.append([f'{context} *{t}'])
                new_term = False
    return terms


def format_xml_match(sents, match_id, context, highlight_query, query_matches):
    """format match in xml style. Returns a filled bs4.element.Tag
    Parameters:
               sents (list):
                     list of sentences (str)
               match_id (int):
                     id of the matched sentence within sents
               context (int):
                     how many sentences of context to display around
                     the matched sentence
               highlight_query (bool):
                     whether or not the query term(s) should be
                     highlighted within the matched sentence
    """
    soup = BeautifulSoup()
    new_matches = []
    if match_id:
        for id_ in match_id:
            new_match = soup.new_tag("match")
            start = max(id_ - context, 0)
            end = min(id_ + context + 1, len(sents))
            new_match.append(" ".join([str(s) for s in sents[start:id_]]))
            # correct for inconsistent tokenisation
            new_match.string = new_match.string
            # add a new tag for the highlighted sentence
            tag = soup.new_tag("em")
            tag.string = ""
            # highlight the query match if needed
            if highlight_query and query_matches:
                q_tags = {}
                tag_order = []
                for q in query_matches:
                    q_tag = soup.new_tag("b")
                    text = str(q)
                    q_tag.append(text)
                    if text in q_tags:
                        q_tags[text].append(q_tag)
                    else:
                        q_tags[text] = [q_tag]
                    tag_order.append(text)
                for token in sents[id_].split():
                    text = str(token)
                    if tag_order and text == tag_order[0]:
                        tag.append(" ")
                        tag.append(q_tags[text].pop(0))
                        tag_order.pop(0)
                        if not q_tags[text]:
                            del q_tags[text]
                    else:
                        if text in ".,;:!?":
                            tag.append(text)
                        else:
                            tag.append(" " + text)
            else:
                tag.string = str(sents[id_])
            new_match.append(tag)
            if end <= len(sents):
                new_match.append(
                    f' {" ".join([str(s) for s in sents[id_+1:end]])}')
            new_matches.append(new_match)
        return new_matches


def format_match(sents, match_id, context, highlight_query):
    """return match as match sentence and lists of left and
    right context sentences
    Parameters:
               sents (list):
                     list of sentences (str)
               match_id (int):
                     id of the matched sentence within sents
               context (int):
                     how many sentences of context to display around
                     the matched sentence
               highlight_query (bool):
                     whether or not the query term(s) should be
                     highlighted within the matched sentence
    """
    if match_id:
        matches = []
        for id_ in match_id:
            start = max(id_ - context, 0)
            end = min(id_ + context + 1, len(sents))
            left_context = [str(s) for s in sents[start:id_]]
            # highlight the query match if needed
            # not sure if this is needed here
            if highlight_query and query_matches:
                pass
            else:
                match = sents[id_]
            if end <= len(sents):
                right_context = [str(s) for s in sents[id_+1:end]]
        assert match is not None,\
            f"No match found for {' '.join([left_context, right_context])}"
        matches.append({"left": left_context,
                        "right": right_context,
                        "match": str(match)})
    return matches


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


def format_output(queries, dest=None):
    """reformat query matches and write them to dest
    Parameter:
              queries (bs4.element.ResultSet):
                      result set of all queries in a document
              dest (BufferedWriter):
                      an opened file to write to (optional)
    """

    new_matches = BeautifulSoup()
    i = 0
    for query in queries:
        matches = query.find_all('match')
        query_exp = [t.split("(")[1].replace('"', '').replace(')', '')
                     .replace('~2', '').replace('OR', '|') for t in terms]

        for match in matches:
            i += 1
            _matches = segment_match(match, query_exp, new_match, True)
            for new_match in _matches:
                new_match.attrs = match.attrs
                if new_match.text:
                    new_matches.append(new_match)

    print(new_matches, file=dest)


def restructure_hit_sample():
    with open("new_format.xml", "w") as ofile:
        print("<xml>", file=ofile)
        format_output(matches, ofile)
        print("</xml>", file=ofile)


def hits_to_txt(queries=queries, remove_non_kw=False, input_files=None):
    """print all matched sentences to text format and save
    context in separate file
    Parameters:
               queries (list):
                      the match objects (organised by query) to print
               remove_non_kw (bool):
                      whether or not the query terms should be matched
                      against the expanded dict again
               input_files (list):
                      list of filenames to process.
    """
    matches = set()
    if input_files:
        queries = set()
        for file in input_files:
            print(file)
            with open(file) as ifile:
                mark_up = BeautifulSoup(
                    re.sub(r'\s+', ' ',
                           re.sub(r'\n', ' ', ifile.read())),
                    features='lxml')
                queries = queries.union(set(mark_up.find_all('query')))

    terms = [re.sub('target:|body:', '', q['term']) for q in queries]
    query_exp = " | ".join(
        [t.replace('"', '').replace(')', '')
         .replace('~2', '').replace('OR', '|')
         for exp in terms for t in [exp.split('(')[-1]]])
    hits = "hit_sample.csv"
    context = "context.csv"
    if remove_non_kw:
        hits = f'filtered_{hits}'
        context = f'filtered_{context}'
    with open(hits, "w") as hit_file:
        hit_writer = csv.writer(hit_file, delimiter=";")
        if remove_non_kw:
            for query in queries:
                q_term = query["term"].split("(")[0].strip('"')
                if query['term'].split("(")[0] in expanded_dict:
                    matches = query.find_all("match")
                    for match in matches:
                        match_data = (q_term + f'_{match["match_nb"]}',
                                      match['doc'].strip("\n"),
                                      match['section'].strip("\n"))
                        if type(match) == dict:
                            match = match['match']
                        segments = segment_match(match,
                                                 query_exp,
                                                 xml=False)
                        for segment in segments:
                            hit_writer.writerow(
                                [*match_data,
                                 ' '.join(segment['left']),
                                 segment['match'].lstrip('\n'),
                                 ' '.join(segment['right'])])

        else:
            for query in queries:
                if '(' in query['term'] and 'doc_title' not in query['term']:
                    parenthesis_format = True
                    q_term = query["term"].split("(")[0].strip('"')
                else:
                    parenthesis_format = False
                matches = query.find_all("match")
                if not matches:
                    matches = set()
                    for sec in query.find_all('section'):
                        matches = matches.union(sec.find_all('match'))
                for i, match in enumerate(matches):
                    if parenthesis_format:
                        match_data = (q_term + f'_{match["match_nb"]}',
                                      match['doc'].strip("\n"),
                                      match['section'].strip("\n"))
                    else:
                        match_data = (match['doc'].strip("\n"),
                                      match['section'].strip("\n"))
                    if type(match) == dict:
                        match = match['match']
                    segments = segment_match(match, query_exp, xml=False)
                    for segment in segments:
                        hit_writer.writerow(
                            [*match_data,
                             ' '.join(segment['left']),
                             segment['match'].lstrip('\n'),
                             ' '.join(segment['right'])])


def compare_boundaries():
    for i, m in enumerate(matches):
        doc = model(m.text)
        s = list(doc.sents)
        sents = redefine_boundaries(doc)
        print(f"{i}:", len(s), len(sents))
        for j in range(max(len(s), len(sents))):
            if j < len(s):
                print("1:", s[j])
            else:
                print("1:")
            if j < len(sents):
                print("2:", sents[j])
            else:
                print("2:")
            print()
        if input("continue? (Y/n)\n> ").casefold() == "n":
            break


def target_search(match, topics):
    for topic in topics:
        if re.search('|'.join(topic), match.b.text):
            if 'class' not in match.attrs:
                match['class'] = []
            match['class'].append(topic[0])


def context_search(match, topics):
    for topic in topics:
        if re.search('|'.join(topic), match.text):
            if 'class' not in match.attrs:
                match['class'] = []
            match['class'].append(topic[0])
            print(match['class'])


def format_pilot_study(files, topics, search_context=True):
    """apply topic filter to search results and output
       the initial hits with the ones matching the filters
       highlighted in colour.

    Parameters:
              files (list):
                    list of filenames with results to merge
                    into a single file. (This was used for
                    document-wise search.
              topics (list):
                    a list of topics, each topic represented
                    by a list of one or more keywords.
              search_context (bool):
                    wether or not to filter based on the target
                    sentence only, or the full context window
    """
    matches = BeautifulSoup('<html><head></head><body></body></html>', parser='html.parser')
    style = matches.new_tag('style')
    colors = ['lightblue', 'lightgreen', 'coral', 'gold', 'plum']
    style.append("""body {
    margin-top: 4%;
    margin-bottom: 8%;
    margin-right: 13%;
    margin-left: 13%;
    }""")

    for i, topic in enumerate(topics):
        style.append(f'.{remove_accent_chars(topic[0])}' +' { background-color:' + colors[i] + ';}')
    matches.head.append(style)
    matches = matches.body
    if search_context:
        search_funct = context_search
        prefix = 'context_'
    else:
        search_funct = target_search
        prefix = 'target_only_'
    last_section = None
    for file in files:
        with open(file) as ifile:
            soup = BeautifulSoup(ifile.read(), parser='html.parser')
            for match in soup.find_all('p'):
                search_funct(match, topics)
                if 'class' in match.attrs:
                    tag = soup.new_tag('div')
                    tag.append(f"Topic: {' '.join(match['class'])}")
                    match['class'] = [remove_accent_chars(c) for c in match['class']]
                    match.append(tag)

                header_text = f"Document {match['doc']}, " +\
                f"Section {match['section']}"
                if last_section and header_text == last_section.h3.text:
                    last_section.append(match)
                else:
                    sec = soup.new_tag('section')
                    header = soup.new_tag('h3')
                    header.append(header_text)
                    sec.append(header)
                    sec.append(match)
                    matches.append(sec)
                    last_section = sec
    
    with open(f'{prefix}combined_topics.html', 'w') as ofile:
        matches = matches.parent
        ofile.write(matches.prettify(formatter='html5'))

# files = glob.glob('incr_decr_docs/*.html')
# topics = [['arbetslöshet'], ['tillväxt'],
# ['klimat'], ['missbruk'], ['hälsa']]

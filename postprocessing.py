from bs4 import BeautifulSoup
# import fasttext
import glob
from keyword_search import redefine_boundaries
from search_terms import expanded_dict
import spacy
import pytest
import copy
import csv
import os
import re
import regex
import unicodedata
from util import find_nearest_neighbour

path = os.path.dirname(os.path.realpath('__file__'))

if not path.startswith('/'):
    path = '/' + path

def fix_file(filename):
    """replace magic characters in a markup file with entity characters"""
    if filename.endswith('.html'):
        markup = 'html'
    elif filename.endswith('.xml'):
        markup = 'xml'
    if not os.path.exists(filename):
        filename = f'{path}/{filename}'
    with open(filename) as ifile:
        soup = BeautifulSoup(ifile.read(), parser=markup,
                             features='lxml')
    segments = filename.split('/')
    dir_, name = '/'.join(segments[:-1]), segments[-1]
    with open('/'.join([dir_, 'new_' + name]) if dir_ else 'new_' + name, 'w') as ofile:
        if markup == 'html':
            ofile.write(soup.prettify(formatter='html5'))
        else:
            ofile.write(soup.prettify())  # formatter='xml'))


def remove_accent_chars(x: str):
    return regex.sub(r'\p{Mn}', '', unicodedata.normalize('NFKD', x))

sample_file = f"{path}/samples/hit_samplereconstructed.xml"
if os.path.exists(sample_file):
    with open(sample_file) as ifile:
        mark_up = BeautifulSoup(ifile.read(), features='lxml')
    # markup
    matches = mark_up.find_all('match')
    queries = mark_up.find_all('query')

    # string representations
    hits = [match.text for match in matches]


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


def _test_lms():
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
        q_ex = [r" *(\b.+\b *)?".join(t) for t in query_matches]
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
            elif query_matches and re.match(query_matches[j], str(sent)):
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
        return format_txt_match(sents, match_id,
                            context, highlight_query)


def separate_query_terms(query_terms, query_exp):
    """reconstruct phrases and single terms within the query.
    The output is a regex containing one word of context between
    two parts of the same phrase if the phrase is segmented.
    """
    terms = []
    new_term = True
    if not query_exp.startswith('|') and not query_exp.endswith('|'):
        query_exp = f'| {query_exp} |'
    for i, term in enumerate(query_terms):
        t = term.text.strip()
        context = ""
        # add previous token to the match string
        previous = model(str(term.previous_sibling))
        if str(previous) not in ["None", " ", ""]:
            previous = str(previous[-1])
            if i > 0 and previous not in [query_terms[i-1], ".", "?", "!"]:
                context = re.sub(r'([\[\])(\\?+*])', r'\\\1', previous)
        if not new_term:
            middle_term = re.sub(r'\(\w+\)\? \*| +\*', ' ',
                                 f'|{"".join(terms[-1])} |'.casefold())
            combined_term = re.sub(r'\(\w+\)\? \*| +\*', ' ',
                                f'{" ".join(terms[-1] + [t])} '.casefold())
            if middle_term in query_exp:
                new_term = True
            elif combined_term in query_exp:
                if context:
                    terms[-1].append(f'({context})? *{t}')
                else:
                    terms[-1].append(f' *{t}')
                continue
        # start of a new term
        if new_term:
            if f'| {t} |'.casefold() in query_exp:
                if context:
                    terms.append([f'({context})? *{t}'])
                else:
                    terms.append([f' *{t}'])
            elif f'| {t} '.casefold() in query_exp:
                if context:
                    terms.append([f'({context})? *{t}'])
                else:
                    terms.append([f' *{t}'])
                new_term = False
    return terms


def format_xml_match(sents, match_id, context, highlight_query,
                     query_matches=[]):
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
    soup = BeautifulSoup(features='lxml')
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
                    text = re.sub(r'[()?*]', '', text).strip()
                    if not text:
                        continue
                    q_tag.append(text)
                    if text in q_tags:
                        q_tags[text].append(q_tag)
                    else:
                        q_tags[text] = [q_tag]
                    tag_order.append(text)
                for token in sents[id_].split():
                    text = str(token)
                    if tag_order and text.strip('.?!:,;') == tag_order[0]:
                        tag.append(" ")
                        tag.append(q_tags[text.strip('.?!:,;')].pop(0))
                        tag_order.pop(0)
                        if text.strip('.?!:,;') != text:
                            tag.append(text[-1])
                        if not q_tags[text.strip('.?!:,;')]:
                            del q_tags[text.strip('.?!:,;')]
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


def format_txt_match(sents, match_id, context, highlight_query):
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


def format_output(queries, dest=None):
    """reformat query matches and write them to dest
    Parameter:
              queries (bs4.element.ResultSet):
                      result set of all queries in a document
              dest (BufferedWriter):
                      an opened file to write to (optional)
    """

    new_matches = BeautifulSoup(features='lxml')
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


def hits_to_txt(queries, remove_non_kw=False, input_files=None):
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
        if re.search('|'.join(topic).replace('.', r'\.'), match.b.text):
            if 'class' not in match.attrs:
                match['class'] = []
            match['class'].append(topic[0])


def context_search(match, topics):
    for topic in topics:
        if re.search('|'.join(topic).replace('.', r'\.'), match.text):
            if 'class' not in match.attrs:
                match['class'] = []
            match['class'].append(topic[0])
            # print(match['class'])


def format_pilot_study(files, topics, search_context=True, prefix=""):
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
    matches = BeautifulSoup('<html><head></head><body></body></html>',
                            parser='html.parser', features='lxml')
    style = matches.new_tag('style')
    colors = ['lightblue', 'lightgreen', 'coral', 'gold', 'plum']
    style.append("""body {
    margin-top: 4%;
    margin-bottom: 8%;
    margin-right: 13%;
    margin-left: 13%;
    }""")
    topics = [[term.replace('##', '\\B') for term in topic] for topic in topics]
    for i, topic in enumerate(topics):
        style.append(f'.{remove_accent_chars(topic[0])}' +
                     ' { background-color:' + colors[i] + ';}')
    matches.head.append(style)
    matches = matches.body
    if search_context:
        search_funct = context_search
        prefix += 'context_'
    else:
        search_funct = target_search
        prefix += 'target_only_'
    last_section = None
    for file in files:
        with open(file) as ifile:
            soup = BeautifulSoup(ifile.read(), parser='html.parser',
                                 features='lxml')
            for match in soup.find_all('p'):
                search_funct(match, topics)
                if 'class' in match.attrs:
                    tag = soup.new_tag('div')
                    tag.append(f"Topic: {' '.join(match['class'])}")
                    match['class'] = [remove_accent_chars(c)
                                      for c in match['class']]
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


def highlight_additions(filename, old_file, output, class_only=True):
    """
    compare all p tags in old and new file and only print the changed ones to
    output.
    """
    def remove_wspace(el):
        return re.sub(r' ([?!.,;:])', r'\1',
                      re.sub('  +', ' ',
                             re.sub('\n', ' ', el.text)))

    def is_topic_match(el):
        return hasattr(el, 'attrs') and 'class' in el.attrs

    with open(filename) as newfile,\
         open(old_file) as oldfile:
        soup = BeautifulSoup(newfile.read(),
                             parser='html.parser',
                             features='lxml')
        new_p = soup.find_all('p')
        if class_only:
            old_p = BeautifulSoup(oldfile.read(),
                                  parser='html.parser',
                                  features='lxml').find_all(
                                      is_topic_match)
        else:
            old_p = BeautifulSoup(oldfile.read(),
                                  parser='html.parser',
                                  features='lxml').find_all('p')

    additions = []
    counter = 0
    for new_paragraph in new_p:
        is_new = True
        for old_paragraph in old_p:
            if remove_wspace(new_paragraph.b) == \
               remove_wspace(old_paragraph.b):
                counter += 1
                if class_only:
                    if new_paragraph['class'] == old_paragraph['class']:
                        is_new = False
                        break
                else:
                    is_new = False
                    break
        if is_new:
            additions.append(new_paragraph)
    print(len(old_p), len(new_p), len(additions))
    matches = BeautifulSoup('<html><head></head><body></body></html>',
                            parser='html.parser', features='lxml')
    style = matches.new_tag('style')
    colors = ['lightblue', 'lightgreen', 'coral', 'gold', 'plum']
    style.append("""body {
    margin-top: 4%;
    margin-bottom: 8%;
    margin-right: 13%;
    margin-left: 13%;
    }""")

    for i, topic in enumerate(terms):
        style.append(f'.{remove_accent_chars(topic[0])}' +
                     ' { background-color:' + colors[i] + ';}')
    matches.head.append(style)
    matches = matches.body
    last_section = None
    for match in additions:
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

    with open(f'{output}.html', 'w') as ofile:
        matches = matches.parent
        ofile.write(matches.prettify(formatter='html5'))


# files = glob.glob('incr_decr_docs/*.html')
BERT_neighbors = {'hälsa': [('hälsa', 0.0), ('Hälsa', 0.37756878),
                            ('hälsar', 0.47435445), ('hälsade', 0.4799701),
                            ('hälsan', 0.48897803), ('hälsas', 0.6012217),
                            ('hälsotillstånd', 0.63329464),
                            ('hälso', 0.6602299), ('ohälsa', 0.67878914),
                            ('Hälso', 0.6915869), ('##häls', 0.6926912),
                            ('##älsa', 0.71705574), ('hälsovård', 0.72128063),
                            ('välbefinnande', 0.72171354),
                            ('folkhäls', 0.73417914), ('välkomna', 0.74378276),
                            ('hälsning', 0.7548151), ('hälsot', 0.7667723),
                            ('##shäls', 0.76739454), ('hälsok', 0.76976)],
                  'tillväxt': [('tillväxt', 0.0), ('tillväxten', 0.28048193),
                               ('expansion', 0.52488095), ('##växt', 0.55630785),
                               ('tillväx', 0.5663575), ('##växten', 0.5789602),
                               ('växa', 0.5883035), ('expansionen', 0.613449),
                               ('växte', 0.6244403), ('växande', 0.6280979),
                               ('växer', 0.6345276), ('ökning', 0.63854784),
                               ('utveckling', 0.6401801), ('utvecklings', 0.6425348),
                               ('##växande', 0.6512682), ('tillbakagång', 0.6517607),
                               ('Ökningen', 0.6528701), ('Utveckling', 0.65303063),
                               ('expansiva', 0.6592035), ('ökningen', 0.6641777)],
                  'klimat': [('klimat', 0.0), ('klimatet', 0.30628943),
                             ('Klimat', 0.36340815), ('##klimat', 0.4992742),
                             ('Klimatet', 0.5150599), ('##sklimat', 0.60756314),
                             ('miljö', 0.6209361), ('växthus', 0.6220983),
                             ('##limat', 0.65388376), ('temperatur', 0.6637261),
                             ('Miljö', 0.66544074), ('miljöm', 0.66620857),
                             ('ekosystem', 0.6713495), ('miljön', 0.68185645),
                             ('atmosfär', 0.68212783), ('köld', 0.69251716),
                             ('väder', 0.6926243), ('miljök', 0.7020524),
                             ('Miljön', 0.7155703), ('varmare', 0.7163896)],
                  'arbetslöshet': [('arbetslöshet', 0.0),
                                   ('arbetslösheten', 0.27266407),
                                   ('Arbetslösheten', 0.3562233),
                                   ('arbetslöshets', 0.42983437),
                                   ('arbetslösa', 0.46224487),
                                   ('arbetslös', 0.48992646),
                                   ('sysselsättnings', 0.5397878),
                                   ('Arbetsmarknad', 0.5528997),
                                   ('arbetsmarknad', 0.5636917),
                                   ('##slöshet', 0.5704941),
                                   ('##slösheten', 0.5725473),
                                   ('kriminalitet', 0.57470983),
                                   ('arbetsmarknads', 0.58911556),
                                   ('##ysselsättning', 0.59747416),
                                   ('lågkonjunktur', 0.6005991),
                                   ('strejker', 0.60614204),
                                   ('inflationen', 0.607628),
                                   ('inflation', 0.60891247),
                                   ('brottslighet', 0.6115984),
                                   ('fattigdomen', 0.6130494)],
                  'missbruk': [('missbruk', 0.0), ('missbruket', 0.3884945),
                               ('##missbruk', 0.4189924), ('##issbruk', 0.45375484),
                               ('missbrukare', 0.5332813), ('knark', 0.6338639),
                               ('narkoman', 0.63879), ('narkomaner', 0.6417084),
                               ('alkoholm', 0.6523485), ('##issbrukare', 0.6625211),
                               ('alkoholis', 0.6655246), ('överdos', 0.6657642),
                               ('droger', 0.6692097), ('Alkohol', 0.66921556),
                               ('narkotikam', 0.6759534), ('återfall', 0.6785668),
                               ('användandet', 0.6790303), ('alkoholen', 0.6793245),
                               ('prostitution', 0.6801337), ('utnyttjade', 0.68087363)]}
terms = [['arbetslöshet'], #['tillväxt'],
         ['klimat'], ['missbruk'], ['hälsa']]
prefix = 'parsed_filtered_BERT'
max_n = 20
BERT_neighbors = {}
for term in terms:
    if type(term) == list:
        term = term[0]
        if term not in BERT_neighbors:
            BERT_neighbors[term] = find_nearest_neighbour(term, max_n)

BERT_topics = [[neighbor for neighbor, similarity in BERT_neighbors[term[0]]]
               for term in terms]
# format_pilot_study(glob.glob('document_search/parsed_experiments/parsed_ix_pos*'), BERT_topics, False, prefix=prefix)
# highlight_additions(f'{prefix}target_only_combined_topics.html',
#                     'target_only_combined_topics.html',
#                     f'document_search/{prefix}additions')
# topics = [['arbetslöshet', 'arbetslöshet', 'arbetslösheten', 'Arbetslösheten'],
#           ['tillväxt', 'tillväxt', 'tillväxten', 'expansion'],
#           ['klimat', 'klimat', 'klimatet', 'Klimat'],
#           ['missbruk', 'missbruk', 'missbruket', '##missbruk'],
#           ['hälsa', 'hälsa', 'Hälsa', 'hälsar']]
# topics = [['arbetslöshet', 'arbetslösheten', 'Arbetslösheten', 'arbetslöshets', 'arbetslösa', 'arbetslös', 'sysselsättnings', 'Arbetsmarknad', 'arbetsmarknad', '##slöshet'],
#           ['tillväxt', 'tillväxten', 'expansion', '##växt', 'tillväx', '##växten', 'växa', 'expansionen', 'växte', 'växande'],
#           ['klimat', 'klimatet', 'Klimat', '##klimat', 'Klimatet', '##sklimat', 'miljö', 'växthus', '##limat', 'temperatur'],
#           ['missbruk', 'missbruket', '##missbruk', '##issbruk', 'missbrukare', 'knark', 'narkoman', 'narkomaner', 'alkoholm', '##issbrukare'],
#           ['hälsa', 'Hälsa', 'hälsar', 'hälsade', 'hälsan', 'hälsas', 'hälsotillstånd', 'hälso', 'ohälsa', 'Hälso']]
# # remove wordpiece segmentation characters
# topics = [[term.replace('##', '\\B') for term in topic] for topic in topics]
# topics = [['arbetslöshet', 'långtidsarbetslöshet', 'massarbetslöshet', 'arbetslöshet.', 'Arbetslöshet', 'arbetslösheten', 'ungdomsarbetslöshet', 'strukturarbetslöshet', 'Massarbetslöshet', 'deltidsarbetslöshet'],
#           ['tillväxt', 'BNPtillväxt', 'tillväxt.', 'jobbtillväxt', 'tillväxtökning', 'sysselsättningstillväxt', 'produktivitetstillväxt', 'tillväxten', 'produktionstillväxt', 'företagstillväxt'],
#           ['klimat', 'klimat.', 'klimatet', 'sommarklimat', 'lokalklimat', 'vinterklimat', 'uteklimat', 'idéklimat', 'klimatOm', 'kontinentalklimat'],
#           ['missbruk', 'missbruk.', 'narkotikamissbruk', 'missbruket', 'opiatmissbruk', 'Missbruk', 'spritmissbruk', 'blandmissbruk', 'cannabismissbruk', 'missbruk-'],
#           ['hälsa', 'hälsa.', 'Hälsa', 'hälsa-', 'hälsa2', 'hälsah', 'allmänhälsa', 'hälsaGod', 'hälsan', 'hälsan.']]


#     ft = fasttext.load_model('fastText/cc.sv.300.bin')
#     n_terms = [5, 10, 15, 20]
#     ft_neighbors = None
#     BERT_neighbors = None
#     ft_prefix = ""
#     BERT_prefix = ""
#     for i in n_terms:
#         max_n = max(n_terms)
#         print(f'expansion parameter: {i}')
#         topics = []
#         # fasttext terms
#         if ft_neighbors is None:
#             ft_neighbors = {}
#         for term in terms:
#             term = term[0]
#             if term not in ft_neighbors:
#                 ft_neighbors[term] = ft.get_nearest_neighbors(term, max_n)
#             topics.append([term] +
#                           [term for sim, term in ft_neighbors[term]][:i])
#         # prefix = f'expanded_fasttext_{i}_'
#         prefix = f'word_b_expanded_fasttext_{i}_'
#         print(topics[0])
#         format_pilot_study(files, topics, False, prefix=prefix)
#         if ft_prefix:
#             highlight_additions(f'{prefix}target_only_combined_topics.html',
#                                 'target_only_combined_topics.html',
#                                 f'document_search/{prefix}additions')
#         else:
#             highlight_additions(f'{prefix}target_only_combined_topics.html',
#                                 f'{ft_prefix}target_only_combined_topics.html',
#                                 f'document_search/{prefix}additions')
#         ft_prefix = prefix

#         # BERT terms
#         topics = []
#         if BERT_neighbors is None:
#             BERT_neighbors = {}
#         for term in terms:
#             if type(term) == list:
#                 term = term[0]
#             if term not in BERT_neighbors:
#                 BERT_neighbors[term] = find_nearest_neighbour(term, max_n)
#             topics.append(BERT_neighbors[term][:i])
#         print(topics[0])
#         # prefix = f'expanded_BERT_{i}_'
#         prefix = f'word_b_expanded_BERT_{i}_'
#         format_pilot_study(files, topics, False, prefix=prefix)
#         if BERT_prefix:
#             highlight_additions(f'{prefix}target_only_combined_topics.html',
#                                 f'{BERT_prefix}target_only_combined_topics.html',
#                                 f'document_search/{prefix}additions')
#         else:
#             highlight_additions(f'{prefix}target_only_combined_topics.html',
#                                 'target_only_combined_topics.html',
#                                 f'document_search/{prefix}additions')
#         BERT_prefix = prefix

#     # SpaCy parsing example
#     width = 20
#     parsed = model('''Vi bedömer att den minskade byråkratin i sig kommer att leda till \
# ett förbättrat djurskydd. En annan ordning skulle enligt domstolen leda till att \
# legitimiteten för bestämmelserna minskade , framför allt mot bakgrund av att det oftast \
# är först vid en ansökan om förlängning som det kan kontrolleras om förutsättningarna \
# varit uppfyllda.''')
#     for token in parsed:
#         print(f"{token.text: <{width}} {token.tag_: <{width}} {token.dep_: <{width}}")
